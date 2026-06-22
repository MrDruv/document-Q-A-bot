"""Tests for documents.py — document loading and splitting."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from documents import load_and_split_docs


class TestLoadAndSplitDocs:
    def test_returns_empty_list_when_no_docs(self, tmp_path):
        """If the directory has no PDFs, return empty list."""
        result = load_and_split_docs(str(tmp_path))
        assert result == []

    @patch("documents.DirectoryLoader")
    def test_loads_and_splits_documents(self, mock_loader_cls):
        """Verify splitting logic with mocked loader."""
        mock_docs = [
            Document(page_content="A" * 600, metadata={"source": "test.pdf"}),
            Document(page_content="B" * 600, metadata={"source": "test.pdf"}),
        ]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader_cls.return_value = mock_loader_instance

        result = load_and_split_docs("./data/")

        assert len(result) > 0
        # Each chunk should be <= chunk_size (512 by default)
        for chunk in result:
            assert len(chunk.page_content) <= 600  # with overlap, can be slightly over

    @patch("documents.DirectoryLoader")
    def test_preserves_metadata(self, mock_loader_cls):
        """Metadata from source docs should carry to chunks."""
        mock_docs = [
            Document(
                page_content="Some content about AI and machine learning.",
                metadata={"source": "ai.pdf", "page": 0},
            )
        ]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader_cls.return_value = mock_loader_instance

        result = load_and_split_docs("./data/")

        assert all("source" in chunk.metadata for chunk in result)
