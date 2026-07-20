import shutil
import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Render mounts Secret Files at /etc/secrets/<filename>, which is READ-ONLY.
# yt-dlp needs to write updated cookies back after each request, so we copy
# the secret file into a writable location once and point yt-dlp at the copy.
SOURCE_COOKIES_PATH = os.getenv("YT_COOKIES_PATH", "/etc/secrets/cookies.txt")
WRITABLE_COOKIES_PATH = "/tmp/cookies.txt"


def _get_writable_cookies_path():
    if not os.path.exists(SOURCE_COOKIES_PATH):
        return None
    # Refresh the writable copy from the secret each call, so edits to the
    # secret file (e.g. re-exporting fresh cookies) take effect on redeploy.
    if not os.path.exists(WRITABLE_COOKIES_PATH):
        shutil.copyfile(SOURCE_COOKIES_PATH, WRITABLE_COOKIES_PATH)
    return WRITABLE_COOKIES_PATH


def download_yt_audio(url: str) -> str:
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
    }

    cookies_path = _get_writable_cookies_path()
    if cookies_path:
        print(f"Using cookies file at {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path
    else:
        print(f"No cookies file found at {SOURCE_COOKIES_PATH} — YouTube may block this as a bot.")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
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