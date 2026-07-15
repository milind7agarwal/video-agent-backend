import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def format_docs(docs):
    formatted = []
    for doc in docs:
        start = doc.metadata.get("start_time", -1.0)
        end = doc.metadata.get("end_time", -1.0)
        if start >= 0 and end >= 0:
            formatted.append(f"[{start:.1f} - {end:.1f}] {doc.page_content}")
        else:
            formatted.append(doc.page_content)
    return "\n\n".join(formatted)

# FIX: Expect segments (list) instead of a raw transcript string
def build_rag_chain(segments: list):
    # Pass the list of segments containing timestamps to the vector store
    vector_store = build_vector_store(segments)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
            based ONLY on the meeting transcript context provided below.

            If the answer is not found in the context, say: 
            "I could not find this information in the meeting transcript."

            Always be concise and precise. If quoting someone, mention it clearly.
            When you provide information from the context, YOU MUST cite the timestamps in your response using the provided format (e.g., "[12.5 - 34.2]").

            Context from meeting transcript:
            {context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    # FIX: Pass the loaded vector_store to get_retriever
    retriever = get_retriever(vector_store)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
            based ONLY on the meeting transcript context provided below.

            If the answer is not found in the context, say: 
            "I could not find this information in the meeting transcript."

            Always be concise and precise. If quoting someone, mention it clearly.
            When you provide information from the context, YOU MUST cite the timestamps in your response using the provided format (e.g., "[12.5 - 34.2]").

            Context from meeting transcript:
            {context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer