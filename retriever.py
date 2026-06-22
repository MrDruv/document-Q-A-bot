# retriever.py
import logging
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import cfg
from text_processing import get_embeddings

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Two-stage retrieval:
      1. Dense retrieval (FAISS) — fast but imprecise
      2. Cross-encoder reranking — slower but much more accurate
    
    Why two stages? Cross-encoders are too slow to run over the whole DB.
    We use them only on the top-k candidates from FAISS.
    """
    def __init__(self, docs):
        if not docs:
            raise ValueError("Cannot build retriever from an empty document list")

        try:
            self.embeddings = get_embeddings()
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding model '{cfg.embed_model}': {exc}") from exc

        try:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        except Exception as exc:
            raise RuntimeError(f"Failed to build FAISS index: {exc}") from exc

        try:
            self.vectorstore.save_local("faiss_index")
        except OSError as exc:
            logger.warning("Could not save FAISS index to disk: %s", exc)
        
        # Stage 1: dense retriever — fetch more than we need
        base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 20,
                "fetch_k": 50,
                "lambda_mult": 0.7
            }
        )
        
        # Stage 2: cross-encoder reranker
        try:
            cross_encoder = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load cross-encoder reranker model: {exc}") from exc

        compressor = CrossEncoderReranker(
            model=cross_encoder,
            top_n=cfg.top_k
        )
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def retrieve(self, query: str) -> list:
        try:
            return self.retriever.invoke(query)
        except Exception as exc:
            logger.error("Retrieval failed for query: %s", exc)
            raise
