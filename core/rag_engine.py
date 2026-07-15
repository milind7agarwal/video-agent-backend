import os
import re

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
except ImportError:  # pragma: no cover - optional dependency
    ChatPromptTemplate = None
    StrOutputParser = None
    RunnablePassthrough = None
    RunnableLambda = None

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:  # pragma: no cover - optional dependency
    ChatMistralAI = None

from core.vector_store import build_vector_store, load_vector_store, get_retriever


def is_lightweight_mode() -> bool:
    value = os.getenv("LIGHTWEIGHT_MODE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None
    if ChatMistralAI is None:
        return None
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
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


class LightweightRAGChain:
    def __init__(self, transcript_segments):
        self.transcript_segments = transcript_segments or []

    def invoke(self, question: str) -> str:
        if not self.transcript_segments:
            return "I could not find this information in the lightweight transcript context."

        query_terms = [term.lower() for term in re.findall(r"\w+", question) if len(term) > 2]
        if not query_terms:
            return "Please ask a more specific question."

        best_segment = None
        best_score = -1

        for segment in self.transcript_segments:
            text = (segment.get("text") or "").lower()
            score = 0
            for term in query_terms:
                if term in text:
                    score += 2
                score += text.count(term)

            if score > best_score:
                best_score = score
                best_segment = segment

        if not best_segment:
            return "I could not find this information in the lightweight transcript context."

        start = best_segment.get("start", 0) or 0
        end = best_segment.get("end", start) or start
        text = (best_segment.get("text") or "").strip()
        if len(text) > 500:
            text = text[:500] + "..."
        return f"[{start:.1f} - {end:.1f}] {text}"


def build_rag_chain(transcript):
    if is_lightweight_mode():
        return LightweightRAGChain(transcript)

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    if llm is None or ChatPromptTemplate is None or StrOutputParser is None:
        return LightweightRAGChain(transcript)

    prompt = ChatPromptTemplate.from_messages(
        [
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
        ]
    )

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    if is_lightweight_mode():
        return LightweightRAGChain([])

    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    if llm is None or retriever is None or ChatPromptTemplate is None or StrOutputParser is None:
        return LightweightRAGChain([])

    prompt = ChatPromptTemplate.from_messages(
        [
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
        ]
    )

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
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