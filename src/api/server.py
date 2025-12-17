"""
FastAPI Server for Audio Customer Support Agent


This module provides REST API endpoints for testing the audio support pipeline.
Students can use this server to test their implementations via HTTP requests.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

from src.pipeline import AudioSupportPipeline, create_pipeline

class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str
    parameters: Optional[Dict[str, Any]] = {}

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]
    message: str

class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int

app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
pipeline: Optional[AudioSupportPipeline] = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@app.on_event("startup")
async def startup_event():
    """
    Initialize the pipeline on server startup with the chosen free/low-cost stack:
    - STT: Whisper local (no API key)
    - LLM: Gemini (GOOGLE_API_KEY)
    - TTS: Edge TTS (no API key)
    """
    global pipeline
    try:
        logger.info("Starting Audio Support Agent API server...")

        stt_config = {
            "model": os.getenv("WHISPER_MODEL", "base"),  # Whisper local model
        }  # UploadFile handling is via FastAPI; STT writes to temp WAV [web:241][web:262]

        llm_config = {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "model": os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        }  # ChatGoogleGenerativeAI expects GOOGLE_API_KEY env [web:251]

        tts_config = {
            "voice": os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural"),
        }  # Edge TTS bytes returned with audio/mpeg response [web:239][web:16]

        pipeline = await create_pipeline(stt_config, llm_config, tts_config, enable_logging=True)
        logger.info("Pipeline initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        # Keep server up for debugging even if pipeline failed to init

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {"message": "Audio Customer Support Agent API", "version": "1.0.0", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint reporting component readiness.
    """
    global pipeline
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={"pipeline_initialized": False, "stt_ready": False, "llm_ready": False, "tts_ready": False},
            message="Pipeline not initialized",
        )
    try:
        components = await pipeline.health_check()  # readiness dict [web:271][web:272]
        all_healthy = all(components.values())
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            components=components,
            message="All components ready" if all_healthy else "Some components not ready",
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(status="error", components={}, message=f"Health check failed: {str(e)}")

@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process a text-only query via LLM and synthesize audio.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        import time
        start = time.time()
        agent_text, audio_bytes = await pipeline.process_text(request.text, **(request.parameters or {}))
        ms = int((time.time() - start) * 1000)
        return TextResponse(response_text=agent_text, audio_available=bool(audio_bytes), processing_time_ms=ms)
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """
    Full pipeline: speech -> text (STT) -> answer (LLM) -> speech (TTS). Returns MP3 bytes.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        data = await audio.read()  # bytes uploaded as multipart/form-data [web:241][web:263]
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        audio_bytes = await pipeline.process_audio(data)  # STT -> LLM -> TTS
        return Response(content=audio_bytes, media_type="audio/mpeg", headers={"Content-Disposition": "attachment; filename=response.mp3"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    TTS-only: synthesize speech from text and return MP3 bytes. Useful for TTS testing.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        audio_bytes = await pipeline.tts.synthesize(text)
        return Response(content=audio_bytes, media_type="audio/mpeg", headers={"Content-Disposition": "attachment; filename=tts_output.mp3"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    STT-only: transcribe an uploaded audio file to text for debugging.
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        text = await pipeline.stt.transcribe(data)  # Whisper transcribe path
        return {"transcription": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
