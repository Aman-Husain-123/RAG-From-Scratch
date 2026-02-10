from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- 1. Load environment variables ----------
# .env should live in this folder or above; adjust path if needed
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file before running this script.")

# ---------- 2. Load and split PDF ----------
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)

documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)
print(f"Number of chunks after splitting: {len(split_docs)}")

# ---------- 3. Embeddings ----------
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# ---------- 4. Vector store (Qdrant) ----------
# One-time creation + ingestion (uncomment if you have NOT created the collection yet):
# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder,
# )
# print("Ingestion done.")

# Normal usage: connect to existing collection and query it
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder,
)

query = "What is FS module?"
relevant_chunks = retriever.similarity_search(query=query, k=4)

# Turn retrieved docs into plain text for the system prompt
context_text = "\n\n---\n\n".join(doc.page_content for doc in relevant_chunks)

SYSTEM_PROMPT = f"""You are a helpful assistant who answers strictly based on the provided context.

Context:
{context_text}
"""

# ---------- 5. Chat completion ----------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=query),
]

response = llm.invoke(messages)
print("\nAnswer:\n", response.content)