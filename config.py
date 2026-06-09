# config.py — centralize all settings, never scatter magic values
from dataclasses import dataclass

@dataclass
class Config:
    # Embeddings
    embed_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.72

    # LLM
    llm_model: str = "llama-3.3-70b-versatile"     # cheap + fast for voice
    temperature: float = 0.2              # low = more factual

    # Voice
    whisper_model: str = "base.en"        # small enough for real-time
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    sample_rate: int = 16000

    # WebSocket
    ws_host: str = "localhost"
    ws_port: int = 8765

cfg = Config()