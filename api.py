# api.py
import os
import re
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from state import make_initial_state
from pipeline import init_pipeline

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB per file
MAX_FILES = 10
MAX_QUESTION_LENGTH = 2000
ALLOWED_EXTENSIONS = {".pdf"}

app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Global state
state = {
    "graph":     None,
    "retriever": None,
    "memory":    None,
    "loaded":    False
}

def _sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters, keeping only the basename."""
    name = Path(filename).name
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name or name.startswith("."):
        name = f"{uuid.uuid4().hex}.pdf"
    return name


@app.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_FILES})")

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    for f in data_dir.glob("*.pdf"):
        try:
            f.unlink(missing_ok=True)
        except PermissionError:
            pass

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed, got '{ext}'")

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)} MB)")

        safe_name = _sanitize_filename(file.filename)
        save_path = data_dir / safe_name
        if not save_path.resolve().is_relative_to(data_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid filename")

        save_path.write_bytes(contents)

    retriever, memory, graph = init_pipeline("./data/")
    state["retriever"] = retriever
    state["memory"]    = memory
    state["graph"]     = graph
    state["loaded"]    = True

    return {"status": "ok"}

@app.post("/ask")
async def ask(question: str = Form(...)):
    if not state["loaded"]:
        return JSONResponse({"error": "No documents loaded"}, status_code=400)

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Question too long (max {MAX_QUESTION_LENGTH} chars)")

    result = state["graph"].invoke(make_initial_state(question))
    return {"answer": result["answer"]}

@app.post("/clear")
async def clear():
    if state["memory"]:
        state["memory"].clear()
    return {"status": "cleared"}

@app.get("/status")
async def status():
    return {"loaded": state["loaded"]}