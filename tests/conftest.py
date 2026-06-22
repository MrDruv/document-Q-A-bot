"""Shared fixtures for document-Q-A-bot tests."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so tests can import modules directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_documents():
    """Create lightweight mock Document objects for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Python is a programming language created by Guido van Rossum.",
            metadata={"source": "python_intro.pdf", "page": 0},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml_basics.pdf", "page": 1},
        ),
        Document(
            page_content="FAISS is a library for efficient similarity search.",
            metadata={"source": "faiss_docs.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def mock_llm():
    """Return a mock ChatGroq that returns a canned response."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="This is a test answer.")
    return mock


@pytest.fixture
def mock_retriever():
    """Return a mock RAGRetriever."""
    from langchain_core.documents import Document

    mock = MagicMock()
    mock.retrieve.return_value = [
        Document(
            page_content="Python was created by Guido van Rossum in 1991.",
            metadata={"source": "history.pdf"},
        )
    ]
    return mock
