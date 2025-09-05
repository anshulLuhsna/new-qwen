# /workspace/main.py

import base64
import io
import os
import tempfile
from typing import Tuple

import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IMPORTANT: avoid heavy imports/initialization at module import time.
# The agent will be created in the FastAPI startup hook.
from voice_agent import QwenVoiceAgent  # noqa: E402

# --- Pydantic Models for Request/Response Validation ---
class TextRequest(BaseModel):
    text: str

class AgentResponse(BaseModel):
    text: str
    audio_base64: str

# --- FastAPI Application ---
app = FastAPI(
    title="Qwen 2.5 Omni Voice Agent API",
    description="An API to interact with the Qwen 2.5 Omni 7B model for speech-in and speech-out.",
    version="1.0.1",
)

# (Optional) Enable CORS if you’ll call this from a web frontend.
if os.environ.get("ENABLE_CORS", "0") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Singleton agent (populated at startup) ---
agent: QwenVoiceAgent | None = None


def _numpy_audio_to_base64(audio_numpy, samplerate: int = 24000) -> str:
    """Converts a NumPy audio array to a Base64-encoded WAV string."""
    buffer = io.BytesIO()
    # Ensure mono float32; reshape(-1) flattens [T,1] or similar
    sf.write(buffer, audio_numpy.reshape(-1), samplerate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _first_text(text_field) -> str:
    """Qwen processor returns a list from batch_decode; normalize to str."""
    if isinstance(text_field, (list, tuple)) and text_field:
        return text_field[0]
    return str(text_field)


@app.on_event("startup")
def _startup_load_agent():
    """
    Load the model ONCE at startup (not at import).
    Honors QWEN_MODEL_DIR env var via voice_agent.py defaults.
    """
    global agent
    if agent is None:
        print("Initializing Qwen Voice Agent (startup)...")
        # You can pass explicit model dir if needed:
        # model_dir = os.environ.get("QWEN_MODEL_DIR", "/workspace/models/Qwen2.5-Omni-7B")
        # agent = QwenVoiceAgent(model_name=model_dir)
        agent = QwenVoiceAgent()
        print("Agent loaded. API is ready to accept requests.")


@app.on_event("shutdown")
def _shutdown():
    # If you ever add resources that need closing, do it here.
    pass


@app.post("/respond_to_text", response_model=AgentResponse)
async def respond_to_text(request: TextRequest):
    """
    Endpoint to get a spoken response from a text input.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is empty.")

        print(f"[POST /respond_to_text] text='{request.text[:80]}'...")
        response_text, response_audio = agent.speak_from_text(request.text)  # type: ignore
        reply_text = _first_text(response_text)
        audio_b64 = _numpy_audio_to_base64(response_audio)
        return AgentResponse(text=reply_text, audio_base64=audio_b64)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing text request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/respond_to_audio", response_model=AgentResponse)
async def respond_to_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint to get a spoken response from an audio file upload.
    """
    try:
        if not audio_file or not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided.")

        print(f"[POST /respond_to_audio] file='{audio_file.filename}'")
        # Use correct suffix (extension) for temp file
        _, ext = os.path.splitext(audio_file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav") as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        try:
            response_text, response_audio = agent.speak_from_audio(tmp_path)  # type: ignore
            reply_text = _first_text(response_text)
            audio_b64 = _numpy_audio_to_base64(response_audio)
        finally:
            # Always clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return AgentResponse(text=reply_text, audio_base64=audio_b64)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing audio request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "ok", "service": "Qwen 2.5 Omni Voice Agent API"}


@app.get("/healthz")
def healthz():
    """Lightweight health check for probes."""
    return {"status": "ok"}
