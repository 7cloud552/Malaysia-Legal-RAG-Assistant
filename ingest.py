import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent / ".env")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
TABLE_NAME = "rag_chunk_800_240"
VECTOR_SIZE = 1536  


engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

try:
    engine.init_vectorstore_table(table_name=TABLE_NAME, vector_size=VECTOR_SIZE)
    print(f"Created table '{TABLE_NAME}'.")
except Exception:
    print(f"Table '{TABLE_NAME}' already exists — skipping init.")

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings_model,
)

loader = DirectoryLoader("./corpus", glob="*.pdf", loader_cls=PyPDFLoader)
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} pages from ./corpus")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=240)
documents = text_splitter.split_documents(raw_documents)
print(f"Split into {len(documents)} chunks (chunk_size=800, chunk_overlap=240)")

doc_ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in documents]
vector_store.add_documents(documents, ids=doc_ids)

print(f"Ingested {len(documents)} chunks into '{TABLE_NAME}' on Supabase.")
