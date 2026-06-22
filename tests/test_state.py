"""Tests for state.py — verify AgentState schema."""
from typing import get_type_hints
from state import AgentState


class TestAgentState:
    def test_has_required_keys(self):
        hints = get_type_hints(AgentState, include_extras=True)
        expected_keys = {
            "question",
            "rewritten_query",
            "documents",
            "chat_history",
            "answer",
            "answer_tokens",
            "hallucination_score",
            "retry_count",
            "audio_bytes",
        }
        assert expected_keys == set(hints.keys())

    def test_can_instantiate_with_all_fields(self):
        state: AgentState = {
            "question": "test",
            "rewritten_query": "test rewritten",
            "documents": [],
            "chat_history": [],
            "answer": "",
            "answer_tokens": [],
            "hallucination_score": 1.0,
            "retry_count": 0,
            "audio_bytes": b"",
        }
        assert state["question"] == "test"
        assert state["retry_count"] == 0
