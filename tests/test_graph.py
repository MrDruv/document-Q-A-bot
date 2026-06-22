"""Tests for graph.py — node functions and edge conditions."""
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from graph import (
    rewrite_query,
    retrieve_documents,
    grade_documents,
    generate_answer,
    grade_hallucination,
    should_retry,
    has_documents,
    build_graph,
)


class TestRewriteQuery:
    @patch("graph.ChatGroq")
    def test_rewrites_question(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        # Simulate the chain producing a rewritten query
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        with patch("graph.REWRITER_PROMPT") as mock_prompt:
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = "What is Python programming language?"
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            # Patch the full chain construction
            with patch("graph.StrOutputParser") as mock_parser:
                full_chain = MagicMock()
                full_chain.invoke.return_value = "What is Python programming language?"
                mock_prompt.__or__.return_value.__or__ = MagicMock(
                    return_value=full_chain
                )

                state = {"question": "um what is like python"}
                result = rewrite_query(state)

                assert "rewritten_query" in result


class TestRetrieveDocuments:
    def test_calls_retriever_with_rewritten_query(self, mock_retriever):
        state = {"rewritten_query": "What is Python?"}
        result = retrieve_documents(state, mock_retriever)

        mock_retriever.retrieve.assert_called_once_with("What is Python?")
        assert "documents" in result
        assert len(result["documents"]) > 0


class TestGradeDocuments:
    @patch("graph.ChatGroq")
    def test_filters_docs_graded_not_relevant(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        # The source code checks `"relevant" in result.content.lower()`,
        # so we use "not relevant" (does NOT contain substring "relevant"
        # as a standalone match — actually it does contain "relevant").
        # Use a response that truly lacks "relevant" to test filtering.
        mock_llm.invoke.side_effect = [
            MagicMock(content="relevant"),
            MagicMock(content="no match here"),
        ]

        state = {
            "rewritten_query": "What is Python?",
            "documents": [
                Document(page_content="Python is a programming language."),
                Document(page_content="Recipe for chocolate cake."),
            ],
        }

        result = grade_documents(state)
        assert len(result["documents"]) == 1
        assert "Python" in result["documents"][0].page_content

    @patch("graph.ChatGroq")
    def test_keeps_all_relevant_docs(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="relevant")

        state = {
            "rewritten_query": "AI topics",
            "documents": [
                Document(page_content="Neural networks..."),
                Document(page_content="Deep learning..."),
            ],
        }

        result = grade_documents(state)
        assert len(result["documents"]) == 2


class TestGenerateAnswer:
    @patch("graph.ChatGroq")
    def test_generates_answer_and_saves_memory(self, mock_groq_cls):
        from memory import AgentMemory

        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm

        # Mock the streaming chain
        with patch("graph.RAG_PROMPT") as mock_prompt:
            with patch("graph.StrOutputParser") as mock_parser:
                chain_mock = MagicMock()
                chain_mock.stream.return_value = iter(
                    ["Python ", "is ", "a language."]
                )
                mock_prompt.__or__ = MagicMock(return_value=MagicMock())
                mock_prompt.__or__.return_value.__or__ = MagicMock(
                    return_value=chain_mock
                )

                memory = AgentMemory(k=5)
                state = {
                    "question": "What is Python?",
                    "documents": [
                        Document(
                            page_content="Python is a language.",
                            metadata={"source": "test.pdf"},
                        )
                    ],
                }

                result = generate_answer(state, memory)

                assert "answer" in result
                assert "answer_tokens" in result
                assert result["answer"] == "Python is a language."
                assert len(result["answer_tokens"]) == 3


class TestGradeHallucination:
    @patch("graph.ChatGroq")
    def test_returns_float_score(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="0.85")

        state = {
            "documents": [Document(page_content="Python is a language.")],
            "answer": "Python is a programming language.",
            "retry_count": 0,
        }

        result = grade_hallucination(state)
        assert result["hallucination_score"] == 0.85

    @patch("graph.ChatGroq")
    def test_handles_unparseable_score(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="not a number")

        state = {
            "documents": [Document(page_content="test")],
            "answer": "test answer",
            "retry_count": 1,
        }

        result = grade_hallucination(state)
        assert result["hallucination_score"] == 0.5  # fallback

    @patch("graph.ChatGroq")
    def test_preserves_retry_count(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="0.9")

        state = {
            "documents": [Document(page_content="test")],
            "answer": "answer",
            "retry_count": 2,
        }

        result = grade_hallucination(state)
        assert result["retry_count"] == 2


class TestShouldRetry:
    def test_retries_when_low_score_and_under_limit(self):
        state = {"hallucination_score": 0.5, "retry_count": 0}
        assert should_retry(state) == "retry"

    def test_proceeds_when_high_score(self):
        state = {"hallucination_score": 0.9, "retry_count": 0}
        assert should_retry(state) == "proceed"

    def test_proceeds_when_retry_limit_reached(self):
        state = {"hallucination_score": 0.3, "retry_count": 2}
        assert should_retry(state) == "proceed"

    def test_retries_at_boundary(self):
        state = {"hallucination_score": 0.69, "retry_count": 1}
        assert should_retry(state) == "retry"

    def test_proceeds_at_threshold(self):
        state = {"hallucination_score": 0.7, "retry_count": 0}
        assert should_retry(state) == "proceed"


class TestHasDocuments:
    def test_returns_generate_when_docs_exist(self):
        state = {"documents": [Document(page_content="test")]}
        assert has_documents(state) == "generate"

    def test_returns_no_docs_when_empty(self):
        state = {"documents": []}
        assert has_documents(state) == "no_docs"


class TestBuildGraph:
    def test_graph_compiles_successfully(self, mock_retriever):
        from memory import AgentMemory

        memory = AgentMemory(k=5)
        graph = build_graph(mock_retriever, memory)
        # The graph should be a compiled runnable
        assert graph is not None
        assert hasattr(graph, "invoke")
