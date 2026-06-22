# api.py
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from documents import load_and_split_docs
from retriever import RAGRetriever
from memory import AgentMemory
from graph import build_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
state = {
    "graph":     None,
    "retriever": None,
    "memory":    None,
    "loaded":    False
}

@app.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...)):
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    for f in data_dir.glob("*.pdf"):
        try:
            f.unlink(missing_ok=True)
        except PermissionError:
            pass

    for file in files:
        save_path = data_dir / file.filename
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    docs = load_and_split_docs("./data/")
    state["retriever"] = RAGRetriever(docs)
    state["memory"]    = AgentMemory(k=5)
    state["graph"]     = build_graph(state["retriever"], state["memory"])
    state["loaded"]    = True

    return {"status": "ok", "chunks": len(docs)}

@app.post("/ask")
async def ask(question: str = Form(...)):
    if not state["loaded"]:
        return {
            "answer": result["answer"],
            "hallucination_score": result.get("hallucination_score", 1.0),
            "retry_count": result.get("retry_count", 0)
        }

    agent_state = {
        "question":            question,
        "retry_count":         0,
        "answer_tokens":       [],
        "documents":           [],
        "hallucination_score": 1.0,
    }

    result = state["graph"].invoke(agent_state)
    return {
        "answer":              result["answer"],
        "hallucination_score": result.get("hallucination_score", 1.0),
        "retry_count":         result.get("retry_count", 0),
        "no_docs":             result.get("hallucination_score") == -1.0
    }

@app.post("/clear")
async def clear():
    if state["memory"]:
        state["memory"].clear()
    return {"status": "cleared"}

@app.get("/status")
async def status():
    return {"loaded": state["loaded"]}