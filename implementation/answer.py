
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    convert_to_messages
)
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv(override=True)

# Database configuration
BASE_DIR = Path(__file__).parent.parent
DB_NAME = str(BASE_DIR / "vector_db")
COLLECTION_NAME = "rag_documents"

RETRIEVAL_K = 10

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant.
You are chatting with a user.

Use the provided context to answer questions.
If the answer is not available in the context, say so.

Context:
{context}
"""

# Connect to the same collection used by ingestion
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

print("Database:", DB_NAME)
print("Collection:", COLLECTION_NAME)
print("Total vectors:", vectorstore._collection.count())

# Configure retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


def fetch_context(question: str) -> list[Document]:
    docs = retriever.invoke(question)
    return docs


def combined_question(
    question: str,
    history: list[dict] | None = None
) -> str:

    history = history or []

    prior = "\n".join(
        m["content"]
        for m in history
        if m["role"] == "user"
    )

    return prior + "\n" + question


def answer_question(
    question: str,
    history: list[dict] | None = None
) -> tuple[str, list[Document]]:

    history = history or []

    combined = combined_question(question, history)

    # Step 1: Retrieve documents
    docs = fetch_context(combined)

    # Step 2: Construct context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    # Step 3: Construct messages
    messages = [
        SystemMessage(content=system_prompt)
    ]

    messages.extend(
        convert_to_messages(history)
    )

    messages.append(
        HumanMessage(content=question)
    )

    # Step 4: Generate answer
    response = llm.invoke(messages)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            block["text"]
            if isinstance(block, dict) and block.get("type") == "text"
            else str(block)
            for block in content
        )

    return str(content), docs
