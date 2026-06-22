# state.py
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator

class AgentState(TypedDict):
    """
    The single source of truth flowing through every node.
    
    Annotated[List, operator.add] means lists are APPENDED
    across node runs, not overwritten — critical for streaming.
    """
    question: str                              # raw transcript
    rewritten_query: str                       # after rewriter node
    documents: List[Document]                 # retrieved docs
    chat_history: List                        # from memory
    answer: str                               # generated answer
    answer_tokens: Annotated[List[str], operator.add]  # stream chunks
    hallucination_score: float                # 0-1, lower = safer
    retry_count: int                          # avoid infinite loops
    audio_bytes: bytes                        # TTS output


def make_initial_state(question: str) -> AgentState:
    """Build the default initial state for a graph invocation."""
    return {
        "question": question,
        "retry_count": 0,
        "answer_tokens": [],
        "documents": [],
        "hallucination_score": 1.0,
    }