# graph.py — the core of the system
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from state import AgentState
from retriever import RAGRetriever
from prompts import RAG_PROMPT, REWRITER_PROMPT
from memory import AgentMemory
from config import cfg

# ─── Node Implementations ───────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:
    """
    Node 1: Rewrite the raw STT transcript for better retrieval.
    STT often includes filler words that hurt embedding similarity.
    """
    llm = ChatGroq(model=cfg.llm_model, temperature=0)
    chain = REWRITER_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]})
    return {"rewritten_query": rewritten}


def retrieve_documents(state: AgentState, retriever: RAGRetriever) -> dict:
    """
    Node 2: Two-stage retrieval using the rewritten query.
    Returns filtered, reranked docs.
    """
    docs = retriever.retrieve(state["rewritten_query"])
    return {"documents": docs}


def grade_documents(state: AgentState) -> dict:
    """
    Node 3: Filter docs below relevance threshold.
    Prevents hallucination from weak matches.
    """
    llm = ChatGroq(model=cfg.llm_model, temperature=0)
    grader_prompt = """Rate if this document is relevant to the question.
    Question: {question}
    Document: {document}
    Return only: 'relevant' or 'irrelevant'"""
    
    filtered = []
    for doc in state["documents"]:
        result = llm.invoke(grader_prompt.format(
            question=state["rewritten_query"],
            document=doc.page_content[:500]  # don't send full doc to grader
        ))
        if "relevant" in result.content.lower():
            filtered.append(doc)
    
    return {"documents": filtered}


def generate_answer(state: AgentState, memory: AgentMemory) -> dict:
    """
    Node 4: Generate a streaming answer using retrieved context.
    Chunks are accumulated in answer_tokens for real-time TTS.
    """
    llm = ChatGroq(
        model=cfg.llm_model,
        temperature=cfg.temperature,
        streaming=True
    )
    
    context = "\n\n".join([
        f"[{doc.metadata.get('source', 'doc')}]\n{doc.page_content}"
        for doc in state["documents"]
    ])
    
    chain = RAG_PROMPT | llm | StrOutputParser()
    
    tokens = []
    for chunk in chain.stream({
        "context": context,
        "chat_history": memory.get_history(),
        "question": state["question"]
    }):
        tokens.append(chunk)
    
    answer = "".join(tokens)
    memory.save_turn(state["question"], answer)
    
    return {
        "answer": answer,
        "answer_tokens": tokens
    }


def grade_hallucination(state: AgentState) -> dict:
    """
    Node 5: Check if the answer is grounded in the documents.
    Returns a score; the edge condition decides whether to retry.
    """
    llm = ChatGroq(model=cfg.llm_model, temperature=0)
    prompt = """Given the context and answer, rate if the answer is fully 
    supported by the context. Score 0.0 (not supported) to 1.0 (fully supported).
    Context: {context}
    Answer: {answer}
    Return only a float like 0.8"""
    
    context = "\n".join([d.page_content[:300] for d in state["documents"]])
    result = llm.invoke(prompt.format(
        context=context, answer=state["answer"]
    ))
    
    try:
        score = float(result.content.strip())
    except ValueError:
        score = 0.5
    
    return {
        "hallucination_score": score,
        "retry_count": state.get("retry_count", 0)
    }


# ─── Edge Conditions ────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    """
    Conditional edge: retry retrieval if hallucination score is low,
    but cap at 2 retries to avoid infinite loops.
    """
    if state["hallucination_score"] < 0.7 and state["retry_count"] < 2:
        return "retry"
    return "proceed"


def has_documents(state: AgentState) -> str:
    """
    Conditional edge: if no relevant docs found, generate a 
    'I don't know' response directly — don't hallucinate.
    """
    return "generate" if state["documents"] else "no_docs"


# ─── Graph Assembly ─────────────────────────────────────────────────

def build_graph(retriever: RAGRetriever, memory: AgentMemory) -> StateGraph:
    """
    Assemble the full LangGraph execution graph.
    Design principle: keep node functions pure where possible,
    inject dependencies via closures.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("rewrite",   rewrite_query)
    graph.add_node("retrieve",  lambda s: retrieve_documents(s, retriever))
    graph.add_node("grade_docs",  grade_documents)
    graph.add_node("generate",  lambda s: generate_answer(s, memory))
    graph.add_node("grade_hallucination", grade_hallucination)
    graph.add_node("no_docs",   lambda s: {
        "answer": "I couldn't find relevant information for that question.",
        "answer_tokens": ["I couldn't find relevant information."]
    })
    
    # Define the flow
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    
    # Conditional: do we have docs?
    graph.add_conditional_edges("grade_docs", has_documents, {
        "generate": "generate",
        "no_docs": "no_docs"
    })
    
    graph.add_edge("generate", "grade_hallucination")
    
    # Conditional: retry or finish?
    graph.add_conditional_edges("grade_hallucination", should_retry, {
        "retry": "retrieve",     # loop back with higher retry count
        "proceed": END
    })
    
    graph.add_edge("no_docs", END)
    
    return graph.compile()