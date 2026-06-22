# chunk_tuner.py — find optimal chunk size empirically
import logging
from langchain_community.vectorstores import FAISS
from text_processing import get_embeddings, get_text_splitter

logger = logging.getLogger(__name__)

def evaluate_chunking(docs, test_queries, chunk_size, chunk_overlap):
    """
    Measure retrieval precision at different chunk sizes.
    Rule of thumb:
    - Factual QA:    256–512 tokens, low overlap (32)
    - Summarization: 1024–2048 tokens, medium overlap (128)
    - Voice agents:  512 tokens — balances context and speed
    """
    if not docs:
        raise ValueError("Cannot evaluate chunking with an empty document list")
    if not test_queries:
        raise ValueError("Cannot evaluate chunking with no test queries")

    chunks = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

    if not chunks:
        logger.warning("Splitting produced zero chunks for chunk_size=%d", chunk_size)
        return {
            "chunk_size": chunk_size,
            "num_chunks": 0,
            "recall@3": 0.0
        }

    try:
        vs = FAISS.from_documents(chunks, get_embeddings())
    except Exception as exc:
        raise RuntimeError(f"Failed to build vector store for evaluation: {exc}") from exc
    
    scores = []
    for query, expected_answer in test_queries:
        try:
            results = vs.similarity_search_with_score(query, k=3)
        except Exception as exc:
            logger.warning("Similarity search failed for query '%s': %s", query, exc)
            scores.append(0)
            continue
        hit = any(expected_answer.lower() in r[0].page_content.lower()
                  for r in results)
        scores.append(int(hit))
    
    return {
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "recall@3": sum(scores) / len(scores)
    }
