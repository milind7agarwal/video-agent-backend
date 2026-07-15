import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question, load_rag_chain

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    source: str
    language: str = "english"

class ChatRequest(BaseModel):
    question: str

# Global variable to hold the rag chain in memory for chat endpoint
rag_chain_instance = None

@app.post("/api/process")
def process_video(req: ProcessRequest):
    global rag_chain_instance
    try:
        print("Starting AI Video Assistant processing")
        chunks = process_input(req.source)
        transcript_data = transcribe_all(chunks, req.language)
        transcript_text = transcript_data["text"]
        transcript_segments = transcript_data["segments"]

        title = generate_title(transcript_text)
        summary = summarize(transcript_text)
        action_items = extract_action_items(transcript_text)
        key_decisions = extract_key_decisions(transcript_text)
        open_questions = extract_questions(transcript_text)
        
        rag_chain_instance = build_rag_chain(transcript_segments)

        return {
            "title": title,
            "transcript": transcript_text,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": key_decisions,
            "open_questions": open_questions,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat(req: ChatRequest):
    global rag_chain_instance
    if not rag_chain_instance:
        try:
            rag_chain_instance = load_rag_chain()
        except Exception:
            raise HTTPException(status_code=400, detail="RAG chain not initialized. Please process a video first.")
            
    try:
        answer = ask_question(rag_chain_instance, req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)