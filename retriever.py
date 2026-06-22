# retriever.py
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import cfg
from text_processing import get_embeddings

class RAGRetriever:
    """
    Two-stage retrieval:
      1. Dense retrieval (FAISS) — fast but imprecise
      2. Cross-encoder reranking — slower but much more accurate
    
    Why two stages? Cross-encoders are too slow to run over the whole DB.
    We use them only on the top-k candidates from FAISS.
    """
    def __init__(self, docs):
        self.embeddings = get_embeddings()
        
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local("faiss_index")
        
        # Stage 1: dense retriever — fetch more than we need
        base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",          # max marginal relevance = diversity
            search_kwargs={
                "k": 20,               # fetch 20, rerank to cfg.top_k
                "fetch_k": 50,
                "lambda_mult": 0.7     # 0=diversity, 1=relevance
            }
        )
        
        # Stage 2: cross-encoder reranker
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        compressor = CrossEncoderReranker(
            model=cross_encoder,
            top_n=cfg.top_k
        )
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def retrieve(self, query: str) -> list:
        return self.retriever.invoke(query)