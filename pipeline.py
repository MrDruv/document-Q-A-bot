# pipeline.py — shared pipeline bootstrap
from documents import load_and_split_docs
from retriever import RAGRetriever
from memory import AgentMemory
from graph import build_graph


def init_pipeline(data_dir: str = "./data/", memory_k: int = 5):
    """Load docs, build retriever + memory + graph in one call.

    Returns ``(retriever, memory, graph)`` so callers that need
    individual components (e.g. the WebSocket server) still have access.
    """
    docs = load_and_split_docs(data_dir)
    retriever = RAGRetriever(docs)
    memory = AgentMemory(k=memory_k)
    graph = build_graph(retriever, memory)
    return retriever, memory, graph
