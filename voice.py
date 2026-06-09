# voice.py
import asyncio
import sounddevice as sd
import numpy as np
import websockets
from faster_whisper import WhisperModel
import pyttsx3
from config import cfg

class VoiceInput:
    """
    Real-time microphone → Whisper STT pipeline.
    Uses Voice Activity Detection (VAD) to automatically
    detect when the user has finished speaking.
    """
    def __init__(self):
        # Load once at startup, not per-request
        self.model = WhisperModel(
            cfg.whisper_model,
            device="cpu",
            compute_type="int8"   # quantized = faster on CPU
        )
        self.sample_rate = cfg.sample_rate
        self.silence_threshold = 0.01
        self.silence_duration = 1.5   # seconds of silence = end of speech
    
    def record_until_silence(self) -> np.ndarray:
        """
        Record audio with automatic silence detection.
        Returns audio when 1.5s of silence is detected.
        """
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        audio_chunks = []
        silence_chunks = 0
        silence_limit = int(self.silence_duration / 0.1)
        
        print("🎙 Listening...")
        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                           dtype='float32') as stream:
            while True:
                chunk, _ = stream.read(chunk_size)
                audio_chunks.append(chunk)
                
                # Simple RMS energy for VAD
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < self.silence_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0
                
                if silence_chunks >= silence_limit and len(audio_chunks) > 10:
                    break
        
        return np.concatenate(audio_chunks, axis=0).flatten()
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text using faster-whisper.
        beam_size=1 is fastest; increase for better accuracy.
        """
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,           # remove long silences in audio
            vad_parameters={"min_silence_duration_ms": 300}
        )
        return " ".join([s.text for s in segments]).strip()


class VoiceOutput:
    """
    Text-to-speech output using pyttsx3.
    Offline, no model download needed, works on Python 3.13.
    Speaks the full answer after generation is complete.
    Rate: 175 wpm — natural speaking pace for a voice agent.
    """
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 1.0)

    def speak(self, text: str):
        """Convert text to speech and play through speaker."""
        self.engine.say(text)
        self.engine.runAndWait()


# WebSocket server for streaming tokens to a frontend
async def token_stream_server(websocket, path, graph, retriever, memory):
    """
    WebSocket handler: receives a question, runs the graph,
    streams answer tokens back to the client in real time.
    """
    async for message in websocket:
        state = {
            "question": message,
            "retry_count": 0,
            "answer_tokens": [],
            "documents": [],
            "hallucination_score": 1.0,
        }
        
        async for event in graph.astream(state):
            # Stream token chunks as they're generated
            if "answer_tokens" in event:
                for token in event["answer_tokens"]:
                    await websocket.send(token)
        
        await websocket.send("[DONE]")  # signal completion

async def start_ws_server(graph, retriever, memory):
    handler = lambda ws, path: token_stream_server(
        ws, path, graph, retriever, memory
    )
    async with websockets.serve(handler, cfg.ws_host, cfg.ws_port):
        print(f"WebSocket server on ws://{cfg.ws_host}:{cfg.ws_port}")
        await asyncio.Future()  # run forever