import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ProcessRequest(BaseModel):
    source: str
    language: str = "english"

class ChatRequest(BaseModel):
    question: str

# Global variable to hold the rag chain in memory for chat endpoint
rag_chain_instance = None


def _run_pipeline(chunks, language: str):
    """Shared pipeline used by both the YouTube-URL and file-upload endpoints."""
    global rag_chain_instance
    transcript_data = transcribe_all(chunks, language)
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


@app.post("/api/process")
def process_video(req: ProcessRequest):
    try:
        print("Starting AI Video Assistant processing (YouTube URL)")
        chunks = process_input(req.source)
        return _run_pipeline(chunks, req.language)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-file")
async def process_video_file(file: UploadFile = File(...), language: str = Form("english")):
    try:
        print(f"Starting AI Video Assistant processing (uploaded file: {file.filename})")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_input(file_path)
        result = _run_pipeline(chunks, language)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the uploaded file — Render's free tier has limited disk.
        if os.path.exists(file_path):
            os.remove(file_path)


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
    uvicorn.run(app, host="0.0.0.0", port=8000)