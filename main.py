# /workspace/main.py

import base64
import io
import soundfile as sf
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from voice_agent import QwenVoiceAgent # Import the agent class from the other file

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
    version="1.0.0"
)

# --- Model Loading (Singleton Pattern) ---
# This is the most important part: load the model only ONCE at startup.
print("Initializing FastAPI application and loading the Qwen Voice Agent...")
agent = QwenVoiceAgent()
print("Agent loaded. API is ready to accept requests.")


def numpy_audio_to_base64(audio_numpy, samplerate=24000):
    """Converts a NumPy audio array to a Base64 encoded WAV string."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_numpy.reshape(-1), samplerate, format='WAV')
    buffer.seek(0)
    b64_audio = base64.b64encode(buffer.read()).decode('utf-8')
    return b64_audio

@app.post("/respond_to_text", response_model=AgentResponse)
async def respond_to_text(request: TextRequest):
    """
    Endpoint to get a spoken response from a text input.
    """
    try:
        print(f"Received text request: '{request.text}'")
        response_text, response_audio = agent.speak_from_text(request.text)
        audio_b64 = numpy_audio_to_base64(response_audio)
        
        return AgentResponse(text=response_text, audio_base64=audio_b64)
    except Exception as e:
        print(f"Error processing text request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/respond_to_audio", response_model=AgentResponse)
async def respond_to_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint to get a spoken response from an audio file upload.
    """
    try:
        print(f"Received audio file: {audio_file.filename}")
        # Save the uploaded file to a temporary location so the agent can access it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        response_text, response_audio = agent.speak_from_audio(tmp_path)
        audio_b64 = numpy_audio_to_base64(response_audio)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return AgentResponse(text=response_text, audio_base64=audio_b64)
    except Exception as e:
        print(f"Error processing audio request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Qwen 2.5 Omni Voice Agent API is running."}
