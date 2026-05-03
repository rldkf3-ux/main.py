from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import tempfile
import os
import json
import httpx
import asyncio
from pathlib import Path

app = FastAPI()

# Allow requests from your Vercel frontend

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_methods=[”*”],
allow_headers=[”*”],
)

GROQ_API_KEY = os.environ.get(“GROQ_API_KEY”, “”)
MODEL = “llama-3.3-70b-versatile”
WHISPER_MODEL = “whisper-large-v3”

NOTES_PROMPT = “”“You are an expert content analyst for social media video content.
Analyze the provided transcript and extract structured notes.
Return ONLY raw valid JSON — no markdown fences, no preamble, no explanation.
{
“title”: “Short descriptive title (max 8 words)”,
“contentType”: “Educational / Motivational / Tutorial / Commentary / Entertainment / Other”,
“mainIdea”: “One clear sentence summarizing the core message”,
“keyPoints”: [“point 1”, “point 2”, “up to 7 key points”],
“actionSteps”: [“step 1”, “if not applicable use empty array”],
“notableQuotes”: [“quote 1”, “if not applicable use empty array”],
“topicsThemes”: [“topic 1”, “topic 2”, “3 to 6 topics”],
“overallNotes”: “2-3 paragraph synthesis: what this content is about, who it is for, and why it matters.”,
“fullTranscript”: “The transcript as provided”
}”””

class URLRequest(BaseModel):
url: str
groq_key: str = “”

@app.get(”/”)
def root():
return {“status”: “Reel Notes API running”}

@app.get(”/health”)
def health():
return {“status”: “ok”}

@app.post(”/process”)
async def process_url(req: URLRequest):
api_key = req.groq_key or GROQ_API_KEY
if not api_key:
raise HTTPException(status_code=400, detail=“No Groq API key provided”)

```
url = req.url.strip()
if not url:
    raise HTTPException(status_code=400, detail="No URL provided")

# ── Step 1: Download audio with yt-dlp ──────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    audio_path = os.path.join(tmpdir, "audio.mp3")

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-x",                          # extract audio only
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", audio_path,
        "--no-playlist",
        url
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not download video. Make sure the URL is public and valid. ({result.stderr[:200]})"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Download timed out. Try a shorter video.")

    # Find the downloaded file (yt-dlp may add extension)
    matches = list(Path(tmpdir).glob("audio*"))
    if not matches:
        raise HTTPException(status_code=500, detail="Download failed — no audio file found.")
    audio_file = matches[0]

    # ── Step 2: Transcribe with Groq Whisper ─────────────────
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            with open(audio_file, "rb") as f:
                files = {"file": (audio_file.name, f, "audio/mpeg")}
                data = {"model": WHISPER_MODEL, "response_format": "text"}
                headers = {"Authorization": f"Bearer {api_key}"}
                whisper_resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )

        if whisper_resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid Groq API key.")
        if whisper_resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit hit. Wait a moment and try again.")
        if whisper_resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {whisper_resp.text[:200]}")

        transcript = whisper_resp.text.strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="No speech detected in this video.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# ── Step 3: Generate notes with Groq Llama ───────────────────
try:
    async with httpx.AsyncClient(timeout=60) as client:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL,
            "temperature": 0.3,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": NOTES_PROMPT},
                {"role": "user", "content": f"Analyze this transcript:\n\n{transcript}"}
            ]
        }
        notes_resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

    if notes_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Notes generation failed: {notes_resp.text[:200]}")

    raw = notes_resp.json()["choices"][0]["message"]["content"].strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    notes = json.loads(clean)
    return {"success": True, "notes": notes, "transcript": transcript}

except json.JSONDecodeError:
    # Return transcript even if notes parsing fails
    return {"success": True, "notes": None, "transcript": transcript, "raw": clean}
except HTTPException:
    raise
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Notes error: {str(e)}")
```
