import os
import yt_dlp
from pydub import AudioSegment

# FIX: Path spelling correction
DOWNLOAD_DIR = '/tmp/downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# DEPLOYMENT TRICK: Tell pydub where to look for local FFmpeg binaries when hosted on Render
LOCAL_FFMPEG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin"))
LOCAL_FFMPEG_PATH = os.path.join(LOCAL_FFMPEG_DIR, "ffmpeg")
LOCAL_FFPROBE_PATH = os.path.join(LOCAL_FFMPEG_DIR, "ffprobe")

if os.path.exists(LOCAL_FFMPEG_PATH):
    AudioSegment.converter = LOCAL_FFMPEG_PATH
if os.path.exists(LOCAL_FFPROBE_PATH):
    AudioSegment.ffprobe = LOCAL_FFPROBE_PATH


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
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # FIX: Safer, robust way to replace whatever extension with .wav
        raw_filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(raw_filename)
        filename = f"{base}.wav"
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