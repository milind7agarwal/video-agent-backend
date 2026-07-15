# AI Video Assistant (RAG + Transcription)

An end-to-end app that:
1. Downloads/ingests a YouTube video (or local audio/video),
2. Transcribes it (Whisper for English, Sarvam STT+translate for “hinglish”),
3. Summarizes the transcript,
4. Extracts action items, key decisions, and open questions,
5. Builds a **RAG** index over transcript segments, and
6. Lets you **chat** with answers that include **clickable timestamp citations**.

---

Screenshots
<img width="1440" height="900" alt="Screenshot 2026-07-14 at 2 21 26 PM" src="https://github.com/user-attachments/assets/37503aa4-d764-498e-aa64-6fc136308d75" />

Working Video


https://github.com/user-attachments/assets/e37bc086-26f1-4204-9318-3a4b848ce20c




## Tech Stack

### Frontend
- React (Vite)
- `react-youtube` for video playback

### Backend
- FastAPI
- LangChain LCEL
- Mistral (via `langchain-mistralai`) for:
  - title generation
  - summary
  - action items / decisions / questions
  - RAG question answering
- Chroma vector store (`chromadb`)
- HuggingFace embeddings: `all-MiniLM-L6-v2`
- Speech-to-text:
  - **OpenAI Whisper (local)** for `language=english`
  - **Sarvam** sync STT+translate for `language=hinglish`

---

## How it works

### 1) Process a video (`POST /api/process`)
Backend pipeline:
- `process_input(source)`
  - If `source` is a URL: downloads audio with `yt-dlp`
  - Else: converts provided media to WAV
  - Chunks audio into segments (default: 10 minutes per chunk)
- `transcribe_all(chunks, language)`
  - English → Whisper segments
  - Hinglish → Sarvam (audio is further split into ≤30s pieces to satisfy API limits)
- `summarize(transcript)`
- `generate_title(transcript)`
- `extract_action_items(transcript)`
- `extract_key_decisions(transcript)`
- `extract_questions(transcript)`
- `build_rag_chain(transcript_segments)`
  - Creates Chroma docs using segment text + metadata `{start_time, end_time}`
  - Retrieval: similarity search with `k=4`

### 2) Ask questions (`POST /api/chat`)
- Loads the persistent Chroma store on demand (first time after processing)
- Retrieves relevant chunks
- Prompts the LLM with **only context from retrieved transcript chunks**
- Enforces citation format: `"[12.5 - 34.2] ..."`
- Frontend renders timestamp badges that jump the YouTube player to that time.

---

## Prerequisites

### System dependencies
- **FFmpeg** installed (required by `pydub` + `yt-dlp` audio extraction)

### Environment variables
Create a `.env` file (place it in `video-agent/` or ensure it’s discoverable by FastAPI; `python-dotenv` loads it).

Minimum expected variables:
- `MISTRAL_API_KEY` (required for LLM calls)

If you want `language=hinglish` transcription:
- `SARVAM_API_KEY` (required for Sarvam STT+translate)

Optional:
- `WHISPER_MODEL` (default: `small`)
- `SARVAM_STT_MODEL` (default: `saaras:v2.5`)

---

## Setup

### 1) Backend
```bash
cd video-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Frontend
```bash
cd frontend
npm install
```

---

## Run

### Backend (FastAPI)
```bash
cd video-agent
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

### Frontend (Vite)
```bash
cd frontend
npm run dev
```

**API URL override (optional):**
- Frontend uses `VITE_API_URL` (defaults to `http://localhost:8000`).

---

## API Reference

### `POST /api/process`
Body:
```json
{
  "source": "https://www.youtube.com/watch?v=...",
  "language": "english"
}
```

Response:
```json
{
  "title": "...",
  "transcript": "...",
  "summary": "...",
  "action_items": "...",
  "key_decisions": "...",
  "open_questions": "..."
}
```

### `POST /api/chat`
Body:
```json
{ "question": "What was the main decision about X?" }
```

Response:
```json
{ "answer": "[12.5 - 34.2] ..." }
```

---

## Frontend Usage
1. Paste a **YouTube URL**
2. Click **Analyze**
3. Use the tabs:
   - Summary
   - Action Items
   - Decisions
   - Questions
4. Ask questions in **Ask Questions**
5. Click timestamp badges in answers to jump to the referenced moment.

---

## Notes / Known Behaviors
- The RAG chain is kept in memory in `video-agent/api.py` after processing; `/api/chat` will error if processing hasn’t happened yet.
- Chroma persists to `vector_db/`.
- Transcript chunks for RAG are created by grouping segments until ~400 characters (see `video-agent/core/vector_store.py`).
- Sarvam transcription slices further into 25-second WAV pieces before calling the Sarvam endpoint.

---

## Project Layout
- `frontend/` — React UI
- `video-agent/` — FastAPI backend + RAG pipeline
  - `api.py` — endpoints
  - `utils/audio_processor.py` — download/convert/chunk
  - `core/transcriber.py` — Whisper/Sarvam transcription
  - `core/summarizer.py` — summary + title
  - `core/extractor.py` — action items/decisions/open questions
  - `core/vector_store.py` — Chroma build/load
  - `core/rag_engine.py` — LCEL RAG chain + citation formatting
- `vector_db/` — persistent Chroma files

---

## Example `.env`
```bash
MISTRAL_API_KEY=your_key
SARVAM_API_KEY=your_key  # only needed for hinglish
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```
