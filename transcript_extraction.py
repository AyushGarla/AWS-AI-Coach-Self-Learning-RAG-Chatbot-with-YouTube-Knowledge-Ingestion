# --------------------------------------------------------------
# transcript_extraction.py  (FINAL — Stable, Clean Transcript Engine)
# --------------------------------------------------------------
# Extracts YouTube transcripts using:
#   1️⃣ yt_dlp subtitles (preferred)
#   2️⃣ Whisper AI fallback (speech-to-text)
#
# Output:
#   transcript.txt  → always clean & ready for NLP.py
#
# Fully compatible with:
# - Versioned AWS KB Vectorstores
# - Transcript Vector Store
# - RAG Agent teach() self-learning loop
# --------------------------------------------------------------

import os
import re
import yt_dlp
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# --------------------------------------------------------------
# Download best-quality audio
# --------------------------------------------------------------
def download_audio(url, out_path="audio.mp3"):
    """Downloads clean MP3 audio using yt_dlp + ffmpeg."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    if os.path.exists(out_path):
        os.remove(out_path)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return out_path if os.path.exists(out_path) else None
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return None


# --------------------------------------------------------------
# Whisper AI Transcription (Fallback)
# --------------------------------------------------------------
def whisper_transcribe(audio_path, language="en"):
    """Transcribe audio using Whisper (small model for stability)."""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-small"

    print("🎧 Loading Whisper model...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=25,   # more stable
        batch_size=6,
        torch_dtype=torch_dtype,
        device=device,
    )

    print("🎙️ Whisper transcribing... please wait.")
    result = asr(audio_path, generate_kwargs={"language": language, "task": "transcribe"})

    return result["text"].strip()


# --------------------------------------------------------------
# Pull subtitles using yt_dlp
# --------------------------------------------------------------
def extract_subtitles(url, lang="en"):
    """Fetch auto or manual subtitles from YouTube (preferred)."""

    # Delete old .vtt files
    for file in os.listdir():
        if file.endswith(".vtt"):
            os.remove(file)

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": "temp.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        vtt_file = next((f for f in os.listdir() if f.endswith(".vtt")), None)
        return vtt_file
    except Exception as e:
        print(f"⚠️ Subtitle extraction error: {e}")
        return None


# --------------------------------------------------------------
# Main transcript extraction
# --------------------------------------------------------------
def get_youtube_transcript(url, lang="en"):
    """
    1️⃣ Try subtitles first
    2️⃣ If missing → Whisper fallback
    3️⃣ Always writes clean 'transcript.txt'
    """
    print("▶️ Trying to fetch YouTube subtitles...")
    vtt_file = extract_subtitles(url, lang)

    # ------------------------------------------------------
    # CASE 1: Subtitles found
    # ------------------------------------------------------
    if vtt_file:
        with open(vtt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        transcript_parts = []
        for line in lines:
            line = line.strip()

            # skip timestamps + metadata + numbering
            if "-->" in line:
                continue
            if line.isdigit():
                continue
            if line.upper() == "WEBVTT":
                continue
            if not line:
                continue

            transcript_parts.append(line)

        transcript = " ".join(transcript_parts)
        transcript = re.sub(r"\s+", " ", transcript).strip()

        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        try:
            os.remove(vtt_file)
        except:
            pass

        print("✅ Transcript saved as 'transcript.txt' (from YouTube subtitles).")
        return "transcript.txt"

    # ------------------------------------------------------
    # CASE 2: Whisper fallback
    # ------------------------------------------------------
    print("ℹ️ No subtitles found → using Whisper AI fallback.")

    audio_path = download_audio(url)
    if not audio_path:
        print("❌ Failed to download audio — cannot transcribe.")
        return None

    try:
        transcript = whisper_transcribe(audio_path, language=lang)

        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        print("✅ Transcript saved as 'transcript.txt' (Whisper AI).")
        return "transcript.txt"

    finally:
        # Clean up audio
        if os.path.exists(audio_path):
            os.remove(audio_path)


# --------------------------------------------------------------
# Script (manual test)
# --------------------------------------------------------------
if __name__ == "__main__":
    url = input("Enter YouTube URL: ").strip()
    get_youtube_transcript(url)
