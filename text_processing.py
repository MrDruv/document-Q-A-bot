# text_processing.py — shared text-splitting and embedding factories
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import cfg

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a configured HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_text_splitter(
    chunk_size: int = cfg.chunk_size,
    chunk_overlap: int = cfg.chunk_overlap,
) -> RecursiveCharacterTextSplitter:
    """Return a configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
