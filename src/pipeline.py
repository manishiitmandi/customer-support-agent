"""
Audio Customer Support Agent Pipeline


This module orchestrates the complete STT -> LLM -> TTS pipeline.
Students should complete the implementation to connect all components.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    
    This class manages the entire audio processing pipeline for customer support.
    Students should complete the implementation to make it fully functional.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False

        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)

    async def initialize(self) -> None:
        """
        Initialize all pipeline components (STT -> LLM -> TTS) and verify readiness.
        """
        try:
            self.logger.info("Initializing Audio Support Pipeline...")

            # STT (Whisper local or configured alternative)
            self.logger.info("Initializing STT service...")
            self.stt = STTService(self.config.stt_config or {})
            await self.stt.initialize()  # Whisper load off event loop [web:213][web:214]

            # LLM Agent (Gemini + ReAct + RAG)
            self.logger.info("Initializing LLM agent...")
            self.llm_agent = CustomerSupportAgent(self.config.llm_config or {})
            await self.llm_agent.initialize()  # ReAct agent + Chroma KB [web:251][web:47]

            # TTS (Edge TTS by default)
            self.logger.info("Initializing TTS service...")
            self.tts = TTSService(self.config.tts_config or {})
            await self.tts.initialize()  # Edge TTS voice setup [web:16][web:10]

            # Verify readiness
            if not (self.stt.is_ready() and self.llm_agent.is_initialized and self.tts.is_ready()):
                raise RuntimeError("Some pipeline components failed to initialize")  # safety

            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully!")
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise

    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        Process audio input through STT -> LLM -> TTS and return synthesized audio bytes.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            # Step 1 - STT
            self.logger.info("Converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)  # Whisper transcribe [web:213][web:214]
            self.logger.info(f"Transcribed text: {text_input!r}")

            if not text_input or not text_input.strip():
                self.logger.info("No speech detected; generating prompt to retry.")
                return await self.tts.synthesize("Sorry, no speech was detected. Please try again.")  # Edge TTS [web:16]

            # Step 2 - LLM Agent
            self.logger.info("Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)  # ReAct agent [web:252][web:254]
            self.logger.info(f"Agent response: {agent_response!r}")

            # Step 3 - TTS
            self.logger.info("Converting response to speech...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)  # Edge TTS synth [web:16]
            self.logger.info("Audio response generated successfully")

            return response_audio
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise

    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        Process text input directly with LLM -> TTS (useful for testing without STT).
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            self.logger.info(f"Processing text query: {text_input!r}")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)  # Agent invoke [web:252]
            response_audio = await self.tts.synthesize(agent_response, **kwargs)  # Edge TTS bytes [web:16]
            return agent_response, response_audio
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, bool]:
        """
        Check the health status of all pipeline components.
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }

    async def cleanup(self) -> None:
        """
        Cleanup all pipeline resources (reverse order).
        """
        self.logger.info("Cleaning up pipeline resources...")
        try:
            if self.tts:
                await self.tts.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.stt:
                await self.stt.cleanup()
        finally:
            self.tts = None
            self.llm_agent = None
            self.stt = None
            self.is_initialized = False
            self.logger.info("Pipeline cleanup completed")


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True,
) -> AudioSupportPipeline:
    """
    Factory function to create and initialize a pipeline.
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging,
    )
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    return pipeline


if __name__ == "__main__":
    async def main():
        # Example minimal configs for local/free stack:
        stt_config = {
            "model": "base",  # Whisper local model
        }
        llm_config = {
            # Gemini via langchain-google-genai; GOOGLE_API_KEY should be set in env
            "model": "gemini-1.5-pro",
            "temperature": 0.7,
        }
        tts_config = {
            "voice": "en-US-JennyNeural",  # Edge TTS default voice
        }

        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        # Simple text test
        text = "Hello! What is your return policy?"
        response_text, response_audio = await pipeline.process_text(text)
        print("Agent:", response_text[:120], "...")
        # Write MP3 to confirm output
        with open("pipeline_test_output.mp3", "wb") as f:
            f.write(response_audio)
        await pipeline.cleanup()
        print("Pipeline example completed.")

    asyncio.run(main())
