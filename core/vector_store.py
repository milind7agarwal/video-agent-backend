import os 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(segments : list)->Chroma:
    print("Building vector Store")

    docs = []
    current_chunk_text = ""
    chunk_start = None
    chunk_end = None
    chunk_index = 0

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        
        if chunk_start is None:
            chunk_start = seg["start"]
        
        current_chunk_text += text + " "
        
        if seg["end"] is not None:
            chunk_end = seg["end"]

        if len(current_chunk_text) >= 400 or seg["start"] is None:
            metadata = {
                'chunk_index': chunk_index,
                'start_time': chunk_start if chunk_start is not None else -1.0,
                'end_time': chunk_end if chunk_end is not None else -1.0
            }
            docs.append(Document(page_content=current_chunk_text.strip(), metadata=metadata))
            chunk_index += 1
            current_chunk_text = ""
            chunk_start = None
            chunk_end = None

    if current_chunk_text.strip():
        metadata = {
            'chunk_index': chunk_index,
            'start_time': chunk_start if chunk_start is not None else -1.0,
            'end_time': chunk_end if chunk_end is not None else -1.0
        }
        docs.append(Document(page_content=current_chunk_text.strip(), metadata=metadata))

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store



def load_vector_store() ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )

