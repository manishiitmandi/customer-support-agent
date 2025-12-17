"""
Base Speech-to-Text (STT) Interface


This module defines the abstract base class for Speech-to-Text implementations.
Students should implement the concrete STT class by inheriting from this base class.


Recommended implementation: Deepgram API (free tier available)
Alternative options: OpenAI Whisper, AssemblyAI, or any other STT service
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import asyncio


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations.
    
    This class defines the interface that all STT implementations must follow.
    Students should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT service.
        
        Args:
            config: Configuration dictionary containing API keys, model settings, etc.
                   Example: {"api_key": "your_api_key", "model": "nova-2"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the STT service (setup API clients, load models, etc.).
        This method should be called before using the STT service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
        
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data as bytes
            **kwargs: Additional parameters specific to the STT implementation
                     (e.g., language, model, formatting options)
        
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the STT service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the STT service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class STTService(BaseSTT):
    """
    Generic STT implementation template.
    
    Students should complete this implementation using their chosen STT service or pretrained model.
    
    API-based options:
    - Deepgram API (free tier, high accuracy): pip install deepgram-sdk
    - AssemblyAI (API-based): pip install assemblyai
    - Azure Speech Services: pip install azure-cognitiveservices-speech
    - Google Cloud Speech: pip install google-cloud-speech
    
    Pretrained model options (local inference):
    - OpenAI Whisper: pip install openai-whisper (various sizes: tiny, base, small, medium, large)
    - Wav2Vec2 models: pip install transformers torch (Facebook's pretrained models)
    - SpeechRecognition + offline engines: pip install SpeechRecognition pocketsphinx
    - Vosk models: pip install vosk (lightweight, supports many languages)
    - Coqui STT: pip install coqui-stt (open-source, pretrained models available)
    
    Input: audio_bytes (bytes) - Raw audio data
    Output: transcribed_text (str) - The text transcription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None  # Whisper model or API client instance
    
    async def initialize(self) -> None:
        """
        Whisper local (default):
        Loads the selected Whisper model off the event loop to avoid blocking.
        """
        import whisper  # Whisper Python package [free, local]
        model_name = self.config.get("model", "base")
        loop = asyncio.get_running_loop()
        # Load model in a worker thread to prevent blocking the event loop
        self.client = await loop.run_in_executor(None, whisper.load_model, model_name)
        self.is_initialized = True
        
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio using the configured STT engine (Whisper local by default).
        Saves bytes to a temporary WAV, then calls model.transcribe in a worker thread.
        """
        if not self.is_ready():
            raise RuntimeError("STT service not initialized")
        
        import tempfile
        loop = asyncio.get_running_loop()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(audio_bytes)
            f.flush()
            result = await loop.run_in_executor(None, self.client.transcribe, f.name)
        return (result or {}).get("text", "").strip()
    
    async def cleanup(self) -> None:
        """
        Release resources and mark service as not initialized.
        """
        self.client = None
        self.is_initialized = False
