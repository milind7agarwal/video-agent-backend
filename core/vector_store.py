import os
import numpy as np
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Simple in-memory store — no chromadb, no torch, no sentence-transformers.
# Good enough for a single transcript held in memory per Render worker.
_STORE = {"docs": [], "vectors": None}


def get_embeddings():
    return MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )


def build_vector_store(transcript: str):
    print("Building vector store (Mistral embeddings, in-memory)")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([d.page_content for d in docs])

    _STORE["docs"] = docs
    _STORE["vectors"] = np.array(vectors, dtype=np.float32)
    return _STORE  # acts as our "vector_store" handle


def load_vector_store():
    # Since this is a single-process in-memory store, "loading" just returns
    # whatever was built most recently in this worker.
    if not _STORE["docs"]:
        raise RuntimeError("No vector store in memory. Process a video first.")
    return _STORE


def get_retriever(vector_store, k: int = 4):
    embeddings = get_embeddings()

    def retrieve(question: str):
        docs = vector_store["docs"]
        vectors = vector_store["vectors"]
        if not docs:
            return []

        q_vec = np.array(embeddings.embed_query(question), dtype=np.float32)
        # cosine similarity
        doc_norms = np.linalg.norm(vectors, axis=1)
        q_norm = np.linalg.norm(q_vec)
        sims = (vectors @ q_vec) / (doc_norms * q_norm + 1e-8)

        top_idx = np.argsort(sims)[::-1][:k]
        return [docs[i] for i in top_idx]

    # Wrap as a LangChain-compatible Runnable so `retriever | RunnableLambda(...)` still works
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(retrieve)