# chunk_tuner.py — find optimal chunk size empirically
from langchain_community.vectorstores import FAISS
from text_processing import get_embeddings, get_text_splitter

def evaluate_chunking(docs, test_queries, chunk_size, chunk_overlap):
    """
    Measure retrieval precision at different chunk sizes.
    Rule of thumb:
    - Factual QA:    256–512 tokens, low overlap (32)
    - Summarization: 1024–2048 tokens, medium overlap (128)
    - Voice agents:  512 tokens — balances context and speed
    """
    chunks = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    
    scores = []
    for query, expected_answer in test_queries:
        results = vs.similarity_search_with_score(query, k=3)
        # Check if expected answer appears in top results
        hit = any(expected_answer.lower() in r[0].page_content.lower()
                  for r in results)
        scores.append(int(hit))
    
    return {
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "recall@3": sum(scores) / len(scores)
    }