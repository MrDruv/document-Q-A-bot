"""Tests for config.py — verify defaults and overrides."""
from config import Config, cfg


class TestConfig:
    def test_default_values(self):
        c = Config()
        assert c.embed_model == "all-MiniLM-L6-v2"
        assert c.chunk_size == 512
        assert c.chunk_overlap == 64
        assert c.top_k == 5
        assert c.score_threshold == 0.72
        assert c.llm_model == "llama-3.3-70b-versatile"
        assert c.temperature == 0.2
        assert c.whisper_model == "base.en"
        assert c.sample_rate == 16000
        assert c.ws_host == "localhost"
        assert c.ws_port == 8765

    def test_custom_values(self):
        c = Config(chunk_size=1024, top_k=10, temperature=0.5)
        assert c.chunk_size == 1024
        assert c.top_k == 10
        assert c.temperature == 0.5
        # other defaults unchanged
        assert c.embed_model == "all-MiniLM-L6-v2"

    def test_module_level_cfg_is_config_instance(self):
        assert isinstance(cfg, Config)

    def test_dataclass_is_mutable(self):
        c = Config()
        c.chunk_size = 256
        assert c.chunk_size == 256
