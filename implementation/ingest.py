
import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent

DB_NAME = str(BASE_DIR / "vector_db")
KNOWLEDGE_BASE = BASE_DIR / "uploaded_documents"

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


def fetch_documents():

    documents = []

    # Iterate through all files in uploaded_documents
    for file_path in KNOWLEDGE_BASE.rglob("*"):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        # Select loader based on file extension
        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))

        elif extension in [".txt", ".md"]:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        print(f"Loading: {file_path.name}")

        file_documents = loader.load()

        # Store original filename in metadata
        for doc in file_documents:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = extension
            doc.metadata["doc_id"] = file_path.name

        documents.extend(file_documents)

    return documents


def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_embeddings(chunks):

    vectorstore = Chroma(
        collection_name="rag_documents",
        persist_directory=DB_NAME,
        embedding_function=embeddings
    )

    # Rebuild this collection before inserting new embeddings
    existing_ids = vectorstore._collection.get(
        include=["metadatas"]
    )["ids"]

    # Step 2: Delete all existing vectors
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
        print(f"Deleted {len(existing_ids)} existing vectors")

    vectorstore.add_documents(chunks)

    print(
        "Successfully stored",
        vectorstore._collection.count(),
        "vectors"
    )

    return vectorstore

def ingest_documents():

    # Step 1: Load all uploaded documents
    documents = fetch_documents()

    if not documents:
        return "No documents found in uploaded_documents."

    # Step 2: Split documents into chunks
    chunks = create_chunks(documents)

    if not chunks:
        return "No chunks generated."

    # Step 3: Generate embeddings and store in ChromaDB
    vectorstore = create_embeddings(chunks)

    if vectorstore is None:
        return "Vectorization failed."

    return (
        f"Vectorization successful!\n"
        f"Documents loaded: {len(documents)}\n"
        f"Chunks embedded: {len(chunks)}\n"
        f"Total vectors in ChromaDB: "
        f"{vectorstore._collection.count()}"
    )

if __name__ == "__main__":

    documents = fetch_documents()

    print(f"Total documents loaded: {len(documents)}")

    chunks = create_chunks(documents)

    print(f"Total chunks created: {len(chunks)}")

    create_embeddings(chunks)

    print("Ingestion complete!")
