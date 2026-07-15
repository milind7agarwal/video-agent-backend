#Actionableitems , decision , questions

import os

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:  # pragma: no cover - optional dependency
    ChatMistralAI = None

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
except ImportError:  # pragma: no cover - optional dependency
    ChatPromptTemplate = None
    StrOutputParser = None
    RunnablePassthrough = None
    RunnableLambda = None


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or ChatMistralAI is None:
        return None
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key=api_key, temperature=0.2)



def build_chain(system_prompt: str):
    llm = get_llm()
    if llm is None or ChatPromptTemplate is None or StrOutputParser is None:
        return None
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    if chain is None:
        return "Action items extraction is unavailable because MISTRAL_API_KEY is not configured."

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    if chain is None:
        return "Key decisions extraction is unavailable because MISTRAL_API_KEY is not configured."

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    if chain is None:
        return "Open questions extraction is unavailable because MISTRAL_API_KEY is not configured."

    return chain.invoke(transcript)