import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Request


import shutil
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
async def process_video(
    file: UploadFile = File(None),
    req: Request | None = None,
):


    global rag_chain_instance
    try:
        print("Starting AI Video Assistant processing")
        
        # 1. Check if a raw video file was uploaded
        if file:
            print(f"Received uploaded file: {file.filename}")
            # Securely save the uploaded file stream into Render's designated /tmp directory
            os.makedirs("/tmp/uploads", exist_ok=True)
            input_source = os.path.join("/tmp/uploads", file.filename)
            
            with open(input_source, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
        # 2. Fall back to a text URL if no file was uploaded
        elif req is not None:
            # Frontend sends JSON: { source, language }
            try:
                body = await req.json()
            except Exception:
                body = {}

            source = body.get("source")
            language = body.get("language") or "english"

            if not source:
                raise HTTPException(status_code=400, detail="No video file or YouTube URL provided.")

            print(f"Received source text/URL (json): {source}")
            input_source = source
        else:
            raise HTTPException(status_code=400, detail="No video file or YouTube URL provided.")



        # Process the local /tmp file path or URL
        chunks = process_input(input_source)
        transcript_data = transcribe_all(chunks, language)
        transcript_text = transcript_data["text"]
        transcript_segments = transcript_data["segments"]

        title = generate_title(transcript_text)
        summary = summarize(transcript_text)
        action_items = extract_action_items(transcript_text)
        key_decisions = extract_key_decisions(transcript_text)
        open_questions = extract_questions(transcript_text)
        
        rag_chain_instance = build_rag_chain(transcript_segments)

        # Cleanup the original raw file from /tmp to save RAM/Disk space
        if file and os.path.exists(input_source):
            os.remove(input_source)

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
        # frontend may send incorrect content-type; accept both JSON and form.
        try:
            question = req.question
        except Exception:
            question = None

        if question is None:
            raise HTTPException(status_code=400, detail="Missing question")

        answer = ask_question(rag_chain_instance, question)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
