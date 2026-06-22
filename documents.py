# documents.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from text_processing import get_text_splitter

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

    chunks = get_text_splitter().split_documents(docs)

    print(f"✅ Split into {len(chunks)} chunks")
    return chunks