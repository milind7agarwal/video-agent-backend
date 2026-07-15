import os
import requests

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover - handles environments where pydub import fails
    AudioSegment = None

try:
    import whisper
except ImportError:  # pragma: no cover - optional dependency
    whisper = None

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")


def is_lightweight_mode() -> bool:
    value = os.getenv("LIGHTWEIGHT_MODE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _use_sarvam() -> bool:
    # Sarvam's STT-translate endpoint transcribes ANY spoken language and
    # returns English text, so it's a valid transcription engine regardless
    # of whether the source audio is English or Hinglish. We use it whenever
    # local Whisper isn't available (lightweight/Render deploy), instead of
    # only for "hinglish" as before.
    return is_lightweight_mode() or whisper is None


_model = None


def load_model():
    global _model

    if whisper is None:
        raise RuntimeError("Whisper is not installed. Install it or enable LIGHTWEIGHT_MODE=true.")

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> list:
    try:
        model = load_model()
        result = model.transcribe(chunk_path, task="transcribe")
        return result["segments"]
    except Exception as exc:
        print(f"Whisper transcription failed: {exc}")
        return [{"start": 0.0, "end": 0.0, "text": f"Transcription unavailable: {exc}"}]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> list:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    Works for English and Hinglish (and most other spoken languages) —
    the API always returns English text.
    """
    if not SARVAM_API_KEY:
        return [{
            "start": 0.0,
            "end": 0.0,
            "text": (
                "SARVAM_API_KEY is not configured on the server, so no real "
                "transcription could be produced. Set SARVAM_API_KEY as an "
                "environment variable and reprocess this video."
            ),
        }]

    if AudioSegment is None:
        return [{"start": 0.0, "end": 0.0, "text": "pydub is unavailable, so transcription could not run."}]

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return [{"start": None, "end": None, "text": full_text.strip()}]


def transcribe_chunk(chunk_path: str, language: str = "english") -> list:
    """
    Route one chunk to Whisper (local) or Sarvam (hosted), depending on
    what's actually available in this deployment — not on the requested
    language. Sarvam handles English audio just fine.
    """
    if _use_sarvam():
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> dict:
    all_segments = []
    full_transcript = ""

    engine = "Sarvam AI (hosted)" if _use_sarvam() else "Whisper (local)"
    print(f"Using {engine} for transcription.")

    for i, chunk_info in enumerate(chunks):
        chunk_path = chunk_info["path"]
        chunk_start_s = chunk_info["start"] / 1000.0

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        segments = transcribe_chunk(chunk_path, language=language)

        for seg in segments:
            if seg["start"] is not None:
                seg["start"] += chunk_start_s
                seg["end"] += chunk_start_s
            all_segments.append(seg)
            full_transcript += seg["text"] + " "

    print("Transcription complete.")

    return {"text": full_transcript.strip(), "segments": all_segments}