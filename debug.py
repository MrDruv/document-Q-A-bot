# debug.py
from langchain_core.tracers import ConsoleCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
import time

def run_with_debug(graph, question: str, verbose: bool = True):
    """
    Run the graph with full tracing.
    Use SqliteSaver to checkpoint state — lets you replay from 
    any node for debugging without re-running the whole graph.
    """
    # Persist graph state to SQLite for debugging
    memory = SqliteSaver.from_conn_string(":memory:")
    graph_with_checkpoint = graph.compile(checkpointer=memory)
    
    config = {
        "configurable": {"thread_id": "debug-session"},
        "callbacks": [ConsoleCallbackHandler()] if verbose else []
    }
    
    state = {
        "question": question,
        "retry_count": 0,
        "answer_tokens": [],
        "documents": [],
        "hallucination_score": 1.0,
    }
    
    start = time.perf_counter()
    result = graph_with_checkpoint.invoke(state, config)
    elapsed = time.perf_counter() - start
    
    print(f"\n{'='*50}")
    print(f"Question:  {question}")
    print(f"Answer:    {result['answer']}")
    print(f"Docs used: {len(result['documents'])}")
    print(f"Halluc. score: {result['hallucination_score']:.2f}")
    print(f"Retries:   {result['retry_count']}")
    print(f"Latency:   {elapsed:.2f}s")
    print(f"{'='*50}")
    
    return result