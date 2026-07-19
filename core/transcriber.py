import os
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

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
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk(chunk_path: str, language: str = "english") -> list:
    """
    Send this chunk to Sarvam. Sarvam sync API only accepts ≤30s audio, so we
    split into 25-second pieces, send each separately, and join the transcripts.

    `language` is currently unused for routing (everything goes through Sarvam),
    but kept as a parameter in case you want to pass a language hint to Sarvam
    later or reintroduce per-language behavior.
    """
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


def transcribe_all(chunks: list, language: str = "english") -> dict:

    all_segments = []
    full_transcript = ""

    print("Using Sarvam AI for transcription.")

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