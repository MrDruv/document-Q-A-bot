# documents.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import cfg

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
    docs = loader.load()

    if not docs:
        print("⚠️ No documents found in data/ folder!")
        return []

    print(f"✅ Loaded {len(docs)} pages from {data_dir}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    print(f"✅ Split into {len(chunks)} chunks")
    return chunks