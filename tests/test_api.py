"""Tests for api.py — FastAPI endpoints."""
import io
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api import app, state as app_state


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset global state before each test."""
    app_state["graph"] = None
    app_state["retriever"] = None
    app_state["memory"] = None
    app_state["loaded"] = False
    yield
    app_state["graph"] = None
    app_state["retriever"] = None
    app_state["memory"] = None
    app_state["loaded"] = False


@pytest.fixture
def client():
    return TestClient(app)


class TestStatusEndpoint:
    def test_status_returns_not_loaded(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json() == {"loaded": False}

    def test_status_returns_loaded(self, client):
        app_state["loaded"] = True
        response = client.get("/status")
        assert response.json() == {"loaded": True}


class TestAskEndpoint:
    def test_ask_without_loaded_docs_returns_400(self, client):
        response = client.post("/ask", data={"question": "What is Python?"})
        assert response.status_code == 400
        assert "No documents loaded" in response.json()["error"]

    def test_ask_with_loaded_docs(self, client):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"answer": "Python is a language."}
        app_state["graph"] = mock_graph
        app_state["loaded"] = True

        response = client.post("/ask", data={"question": "What is Python?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Python is a language."
        mock_graph.invoke.assert_called_once()


class TestClearEndpoint:
    def test_clear_without_memory(self, client):
        response = client.post("/clear")
        assert response.status_code == 200
        assert response.json() == {"status": "cleared"}

    def test_clear_with_memory(self, client):
        mock_memory = MagicMock()
        app_state["memory"] = mock_memory

        response = client.post("/clear")
        assert response.status_code == 200
        mock_memory.clear.assert_called_once()


class TestUploadEndpoint:
    @patch("api.build_graph")
    @patch("api.RAGRetriever")
    @patch("api.load_and_split_docs")
    def test_upload_processes_pdf(
        self, mock_load, mock_retriever_cls, mock_build_graph, client, tmp_path
    ):
        mock_load.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_retriever_cls.return_value = MagicMock()
        mock_build_graph.return_value = MagicMock()

        # Create a fake PDF file
        pdf_content = b"%PDF-1.4 fake content"
        response = client.post(
            "/upload",
            files=[("files", ("test.pdf", io.BytesIO(pdf_content), "application/pdf"))],
        )

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["chunks"] == 3
        assert app_state["loaded"] is True

    @patch("api.build_graph")
    @patch("api.RAGRetriever")
    @patch("api.load_and_split_docs")
    def test_upload_multiple_files(
        self, mock_load, mock_retriever_cls, mock_build_graph, client
    ):
        mock_load.return_value = ["c1", "c2"]
        mock_retriever_cls.return_value = MagicMock()
        mock_build_graph.return_value = MagicMock()

        files = [
            ("files", ("a.pdf", io.BytesIO(b"%PDF"), "application/pdf")),
            ("files", ("b.pdf", io.BytesIO(b"%PDF"), "application/pdf")),
        ]
        response = client.post("/upload", files=files)

        assert response.status_code == 200
        assert app_state["loaded"] is True
