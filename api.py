# api.py
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from state import make_initial_state
from pipeline import init_pipeline

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