# documents.py
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from text_processing import get_text_splitter

logger = logging.getLogger(__name__)

def load_and_split_docs(data_dir: str):
    """
    Load all PDFs from the data/ folder
    and split them into chunks.
    """
    # Load all PDFs from the folder
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    try:
        docs = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load documents from {data_dir}: {exc}") from exc

    if not docs:
        logger.warning("No documents found in %s", data_dir)
        return []

    logger.info("Loaded %d pages from %s", len(docs), data_dir)

    try:
        chunks = get_text_splitter().split_documents(docs)
    except Exception as exc:
        raise RuntimeError(f"Failed to split documents into chunks: {exc}") from exc

    logger.info("Split into %d chunks", len(chunks))
    return chunks
