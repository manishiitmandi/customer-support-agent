"""
Base Text-to-Speech (TTS) Interface


This module defines the abstract base class for Text-to-Speech implementations.
Students should implement the concrete TTS class by inheriting from this base class.


Recommended implementation: ElevenLabs API (free tier available)
Alternative options: OpenTTS, gTTS, Azure Speech Services, or any other TTS service
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import io
import asyncio


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        pass
    
    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        pass
    
    def is_ready(self) -> bool:
        return self.is_initialized


class TTSService(BaseTTS):
    """
    Generic TTS implementation template.
    
    This concrete implementation uses Edge TTS (free Microsoft voices) by default.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.voice: str = self.config.get("voice", "en-US-JennyNeural")  # default voice
    
    async def initialize(self) -> None:
        """
        Edge TTS requires no API client; just set the voice and mark ready.
        """
        # Allow override from config
        self.voice = self.config.get("voice", self.voice)
        self.is_initialized = True  # Edge TTS ready immediately [web:16][web:10]
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Synthesize full audio and return MP3 bytes using edge_tts.Communicate.stream().
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        import edge_tts

        # Per-call overrides
        voice = kwargs.get("voice", self.voice)
        rate = kwargs.get("rate")      # e.g., "+10%" or "-5%"
        pitch = kwargs.get("pitch")    # e.g., "+0Hz" or "+2st"
        volume = kwargs.get("volume")  # e.g., "+0%" or "-5%"

        # If prosody controls are provided, wrap text in simple SSML prosody
        payload = text
        if any([rate, pitch, volume]):
            attrs = []
            if rate: attrs.append(f'rate="{rate}"')
            if pitch: attrs.append(f'pitch="{pitch}"')
            if volume: attrs.append(f'volume="{volume}"')
            attr_str = " ".join(attrs)
            payload = f'<speak><prosody {attr_str}>{text}</prosody></speak>'

        communicate = edge_tts.Communicate(payload, voice)
        audio = b""
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio += chunk["data"]
        return audio  # MP3 bytes compatible with audio/mpeg responses [web:16][web:10]
    
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        Produce audio incrementally and return a BytesIO positioned at start.
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        import edge_tts

        voice = kwargs.get("voice", self.voice)
        rate = kwargs.get("rate")
        pitch = kwargs.get("pitch")
        volume = kwargs.get("volume")

        payload = text
        if any([rate, pitch, volume]):
            attrs = []
            if rate: attrs.append(f'rate="{rate}"')
            if pitch: attrs.append(f'pitch="{pitch}"')
            if volume: attrs.append(f'volume="{volume}"')
            payload = f'<speak><prosody {" ".join(attrs)}>{text}</prosody></speak>'

        communicate = edge_tts.Communicate(payload, voice)
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                buffer.write(chunk["data"])
        buffer.seek(0)
        return buffer  # Ready for StreamingResponse or file write [web:16][web:239]
    
    async def cleanup(self) -> None:
        """
        Reset internal state; Edge TTS has no persistent connections to close.
        """
        self.is_initialized = False
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Edge TTS Python library doesn’t expose an official voice-listing API.
        Return a minimal curated list or allow override via config.
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        # Minimal, commonly used voices; extend as needed
        voices = [
            {"voice": "en-US-JennyNeural", "locale": "en-US", "gender": "Female"},
            {"voice": "en-US-GuyNeural", "locale": "en-US", "gender": "Male"},
            {"voice": "en-GB-LibbyNeural", "locale": "en-GB", "gender": "Female"},
            {"voice": "en-IN-NeerjaNeural", "locale": "en-IN", "gender": "Female"},
        ]
        return voices
