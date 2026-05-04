"""edge-tts streaming TTS engine with interruption support."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional
import logging

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4096  # bytes per TTS chunk


class EdgeTTS:
    def __init__(self, default_voice: str = "en-US-AriaNeural", default_rate: str = "+0%") -> None:
        self._default_voice = default_voice
        self._default_rate = default_rate

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: str = "+0Hz",
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return

        voice = voice or self._default_voice
        rate = rate or self._default_rate

        try:
            import edge_tts
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )

            buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.extend(chunk["data"])
                    # Stream in chunks to reduce latency
                    while len(buffer) >= CHUNK_SIZE:
                        yield bytes(buffer[:CHUNK_SIZE])
                        buffer = buffer[CHUNK_SIZE:]

            if buffer:
                yield bytes(buffer)

        except ImportError:
            # Stub for development
            logger.warning("edge-tts not installed — yielding stub audio")
            yield b"\x00" * 1024

        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise

    async def synthesize_full(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> bytes:
        chunks = []
        async for chunk in self.synthesize_stream(text, voice=voice, rate=rate):
            chunks.append(chunk)
        return b"".join(chunks)
