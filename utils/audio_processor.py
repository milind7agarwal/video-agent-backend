import os
import shutil

try:
    import yt_dlp
except ImportError:  # pragma: no cover - optional dependency
    yt_dlp = None

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover - handles environments where pydub import fails
    AudioSegment = None

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# render_build.sh installs a static ffmpeg/ffprobe binary into ~/.local/bin.
# We point pydub + yt-dlp at that path explicitly instead of relying on PATH,
# since PATH isn't always propagated the same way to the uvicorn process on Render.
_LOCAL_BIN = os.path.expanduser("~/.local/bin")
FFMPEG_PATH = shutil.which("ffmpeg") or os.path.join(_LOCAL_BIN, "ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe") or os.path.join(_LOCAL_BIN, "ffprobe")

if AudioSegment is not None and os.path.exists(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH


def download_yt_audio(url: str) -> str:
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Install backend dependencies first.")

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "extractor_args": {"youtube": ["player_client=android"]},
    }
    
    # If the user has provided a cookies.txt file in the root directory, use it
    if os.path.exists("cookies.txt"):
        ydl_opts["cookiefile"] = "cookies.txt"
    # Point yt-dlp straight at the ffmpeg binary dir so its postprocessor
    # doesn't depend on PATH being set correctly in the Render environment.
    if os.path.isdir(_LOCAL_BIN):
        ydl_opts["ffmpeg_location"] = _LOCAL_BIN

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    if AudioSegment is None:
        raise RuntimeError("pydub is not available in this environment.")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    if AudioSegment is None:
        raise RuntimeError("pydub is not available in this environment.")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append({"path": chunk_path, "start": start})
    return chunks


def process_input(source: str) -> list:
    """
    Download/convert/chunk the source into WAV pieces we can transcribe.

    NOTE: this step is NOT heavy (no torch, no ML model loading) — it's just
    yt-dlp + ffmpeg + pydub, all of which are in requirements.render.txt.
    It must always run for real, in both local and lightweight/Render mode.
    LIGHTWEIGHT_MODE only affects which transcription engine is used
    afterwards (see core/transcriber.py), not this step.
    """
    if AudioSegment is None or yt_dlp is None:
        # True last-resort fallback: dependencies genuinely missing.
        print("Audio dependencies unavailable. Skipping preprocessing.")
        return [{"path": source, "start": 0}]

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_yt_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks