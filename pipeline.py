from typing import TypedDict

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import os
from pathlib import Path    
from dotenv import load_dotenv 

load_dotenv(Path(__file__).resolve().parent / ".env")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
TABLE_NAME = "rag_chunk_800_240"
VECTOR_SIZE = 1536 # match with text-embedding-3-small

engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings_model,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-5.6-luna")

class GeneratedQueries(BaseModel):
    queries: list[str] = Field(description="Search queries related to the question")

query_llm = llm.with_structured_output(GeneratedQueries)

prompt_rag_fusion = """
You are a helpful assistant that generates multiple search queries based on a single input query.

Generate multiple search queries related to:

{question}

If the question concerns a general rule that a schedule, exception, or
exclusion might qualify, limit, or override, include one query aimed at
that angle specifically. Otherwise, focus all queries on the question
as asked.

Output exactly 4 search queries.
"""

class GraphState(TypedDict):
    question: str
    generated_queries: list[str]
    retrieved_documents: list[list[Document]]
    fused_documents: list[Document]
    answer: str

def generate_queries(state: GraphState):
    question = state['question']
    prompt = prompt_rag_fusion.format(question=question)
    result = query_llm.invoke(prompt)
    queries = [question] + result.queries

    return {"generated_queries": queries}

def retrieve_documents(state: GraphState):
    queries = state['generated_queries']

    results = retriever.batch(queries)
    return {"retrieved_documents": results}

def reciprocal_rank_fusion(results:list[list[Document]], k: int=60):
    """
    Merge multiple ranked document lists using
    Reciprocal Rank Fusion.
    """

    fused_scores = {}
    documents = {}

    for docs in results:

        for rank, doc in enumerate(docs):

            doc_str = doc.page_content

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc

            fused_scores[doc_str] += 1 / (rank + k)

    reranked_doc_strs = sorted(
        fused_scores,
        key=lambda d: fused_scores[d],
        reverse=True
    )

    return [
        documents[doc_str]
        for doc_str in reranked_doc_strs
    ]

def fuse_documents(state: GraphState):
    results = state['retrieved_documents']
    fused_docs = reciprocal_rank_fusion(results)

    return {"fused_documents": fused_docs}


answer_prompt = """
Answer the question based only on the following context. Each document
below is numbered — cite the documents you used inline like [1], [2].

Context:
{context}

Question:
{question}

Two rules for when the context doesn't state the answer directly:
1. If the context scopes a rule to one category (e.g. "public companies
   must...", or an explicit list of who is excluded), and the question
   asks about a different but related category, use that scoping to
   answer directly. Don't hedge into "the context doesn't address this"
   when the scoping itself is the answer — a rule stated for one group,
   with no exception given, implies it does not apply to a group left out.
2. If the context does not state anything that actually resolves the
   question — even if it discusses a related or nearby topic — say
   plainly that the provided documents do not address this question.
   Discussing a related topic is not the same as answering this one;
   do not construct a plausible-sounding answer from adjacent material.
"""

def generate_answer(state: GraphState):
    question = state['question']
    docs = state['fused_documents']

    # convert documents into text
    context =  "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))

    prompt = answer_prompt.format(context=context, question=question)
    response = llm.invoke(prompt)

    return {"answer": response.content}

builder = StateGraph(GraphState)

builder.add_node("generate_queries",generate_queries)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("fuse_documents", fuse_documents)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "retrieve_documents")
builder.add_edge("retrieve_documents", "fuse_documents")
builder.add_edge("fuse_documents", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()