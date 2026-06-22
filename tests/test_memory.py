"""Tests for memory.py — conversation buffer window memory."""
from memory import AgentMemory


class TestAgentMemory:
    def test_initial_history_is_empty(self):
        mem = AgentMemory(k=5)
        assert mem.get_history() == []

    def test_save_and_retrieve_turn(self):
        mem = AgentMemory(k=5)
        mem.save_turn("What is Python?", "A programming language.")
        history = mem.get_history()
        assert len(history) == 2  # HumanMessage + AIMessage

    def test_window_evicts_old_turns(self):
        mem = AgentMemory(k=2)
        mem.save_turn("Q1", "A1")
        mem.save_turn("Q2", "A2")
        mem.save_turn("Q3", "A3")

        history = mem.get_history()
        # k=2 means last 2 turns kept (4 messages: 2 human + 2 AI)
        assert len(history) == 4
        # First turn should have been evicted
        contents = [m.content for m in history]
        assert "Q1" not in contents
        assert "Q3" in contents

    def test_clear_resets_history(self):
        mem = AgentMemory(k=5)
        mem.save_turn("Q1", "A1")
        mem.clear()
        assert mem.get_history() == []

    def test_custom_k_value(self):
        mem = AgentMemory(k=1)
        mem.save_turn("Q1", "A1")
        mem.save_turn("Q2", "A2")
        history = mem.get_history()
        # Only last turn should remain
        assert len(history) == 2
        contents = [m.content for m in history]
        assert "Q1" not in contents
        assert "Q2" in contents
