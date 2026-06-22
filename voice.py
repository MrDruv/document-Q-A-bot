# voice.py
import asyncio
import logging
import sounddevice as sd
import numpy as np
import websockets
from faster_whisper import WhisperModel
import pyttsx3
from config import cfg
from state import make_initial_state

logger = logging.getLogger(__name__)

class VoiceInput:
    """
    Real-time microphone → Whisper STT pipeline.
    Uses Voice Activity Detection (VAD) to automatically
    detect when the user has finished speaking.
    """
    def __init__(self):
        try:
            self.model = WhisperModel(
                cfg.whisper_model,
                device="cpu",
                compute_type="int8"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load Whisper model '{cfg.whisper_model}': {exc}") from exc
        self.sample_rate = cfg.sample_rate
        self.silence_threshold = 0.01
        self.silence_duration = 1.5
    
    def record_until_silence(self) -> np.ndarray:
        """
        Record audio with automatic silence detection.
        Returns audio when 1.5s of silence is detected.
        """
        chunk_size = int(self.sample_rate * 0.1)
        audio_chunks = []
        silence_chunks = 0
        silence_limit = int(self.silence_duration / 0.1)
        
        logger.info("Listening...")
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                               dtype='float32') as stream:
                while True:
                    chunk, _ = stream.read(chunk_size)
                    audio_chunks.append(chunk)
                    
                    rms = np.sqrt(np.mean(chunk ** 2))
                    if rms < self.silence_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    
                    if silence_chunks >= silence_limit and len(audio_chunks) > 10:
                        break
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Audio recording failed: {exc}") from exc
        
        return np.concatenate(audio_chunks, axis=0).flatten()
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text using faster-whisper.
        beam_size=1 is fastest; increase for better accuracy.
        """
        try:
            segments, _ = self.model.transcribe(
                audio,
                beam_size=1,
                language="en",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300}
            )
            return " ".join([s.text for s in segments]).strip()
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            raise RuntimeError(f"Transcription failed: {exc}") from exc


class VoiceOutput:
    """
    Text-to-speech output using pyttsx3.
    Offline, no model download needed, works on Python 3.13.
    Speaks the full answer after generation is complete.
    Rate: 175 wpm — natural speaking pace for a voice agent.
    """
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize TTS engine: {exc}") from exc
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 1.0)

    def speak(self, text: str):
        """Convert text to speech and play through speaker."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:
            logger.error("TTS playback failed: %s", exc)
            raise RuntimeError(f"TTS playback failed: {exc}") from exc


# WebSocket server for streaming tokens to a frontend
MAX_WS_MESSAGE_LENGTH = 2000

async def token_stream_server(websocket, path, graph, retriever, memory):
    """
    WebSocket handler: receives a question, runs the graph,
    streams answer tokens back to the client in real time.
    """
    async for message in websocket:
        if not isinstance(message, str) or len(message) > MAX_WS_MESSAGE_LENGTH:
            await websocket.send("[ERROR] Message too long or invalid")
            continue

        question = message.strip()
        if not question:
            await websocket.send("[ERROR] Empty message")
            continue

        try:
            async for event in graph.astream(make_initial_state(question)):
                if "answer_tokens" in event:
                    for token in event["answer_tokens"]:
                        await websocket.send(token)
        except websockets.ConnectionClosed:
            logger.warning("WebSocket client disconnected during streaming")
            return
        except Exception as exc:
            logger.error("Error during graph streaming over WebSocket: %s", exc)
            try:
                await websocket.send(f"[ERROR] {exc}")
            except websockets.ConnectionClosed:
                return
            continue
        
        try:
            await websocket.send("[DONE]")
        except websockets.ConnectionClosed:
            logger.warning("WebSocket client disconnected before completion signal")
            return

async def start_ws_server(graph, retriever, memory):
    handler = lambda ws, path: token_stream_server(
        ws, path, graph, retriever, memory
    )
    try:
        async with websockets.serve(handler, cfg.ws_host, cfg.ws_port):
            logger.info("WebSocket server on ws://%s:%s", cfg.ws_host, cfg.ws_port)
            await asyncio.Future()
    except OSError as exc:
        raise RuntimeError(
            f"Failed to start WebSocket server on {cfg.ws_host}:{cfg.ws_port}: {exc}"
        ) from exc
