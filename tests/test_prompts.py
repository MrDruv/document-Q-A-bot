"""Tests for prompts.py — verify prompt templates render correctly."""
from prompts import RAG_PROMPT, REWRITER_PROMPT


class TestRAGPrompt:
    def test_rag_prompt_has_required_variables(self):
        input_vars = RAG_PROMPT.input_variables
        assert "context" in input_vars
        assert "question" in input_vars

    def test_rag_prompt_renders_with_context(self):
        messages = RAG_PROMPT.format_messages(
            context="Python is a language.",
            chat_history=[],
            question="What is Python?",
        )
        # Should produce system + human messages
        assert len(messages) >= 2
        assert "Python is a language." in messages[0].content
        assert "What is Python?" in messages[-1].content

    def test_rag_prompt_system_includes_rules(self):
        messages = RAG_PROMPT.format_messages(
            context="test", chat_history=[], question="test"
        )
        system_msg = messages[0].content
        assert "3 sentences" in system_msg
        assert "I don't have that information" in system_msg


class TestRewriterPrompt:
    def test_rewriter_has_question_variable(self):
        assert "question" in REWRITER_PROMPT.input_variables

    def test_rewriter_renders(self):
        result = REWRITER_PROMPT.format(question="um what is like python")
        assert "um what is like python" in result
        assert "Rewritten query:" in result
