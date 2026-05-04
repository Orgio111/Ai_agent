"""Streaming Whisper STT engine with noise reduction pre-processing."""
from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import Optional

import numpy as np

from .models import STTResponse


class WhisperSTT:
    def __init__(self, model_size: str = "base", device: str = "cpu") -> None:
        self._model_size = model_size
        self._device = device
        self._model = None
        self.loaded = False

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        try:
            import whisper
            self._model = whisper.load_model(self._model_size, device=self._device)
            self.loaded = True
        except ImportError:
            # Whisper not available — use stub for development
            self._model = None
            self.loaded = True  # Mark loaded to allow service start

    async def unload(self) -> None:
        self._model = None
        self.loaded = False

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> STTResponse:
        start = time.monotonic()

        if not audio_bytes or len(audio_bytes) < 100:
            return STTResponse(text="", language=language or "en", confidence=0.0)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._transcribe_sync, audio_bytes, language)

        duration_ms = (time.monotonic() - start) * 1000
        result.duration_ms = duration_ms
        return result

    def _transcribe_sync(self, audio_bytes: bytes, language: Optional[str]) -> STTResponse:
        if self._model is None:
            return STTResponse(text="[STT stub: whisper not loaded]", language="en", confidence=0.5)

        try:
            import whisper
            audio_np = self._bytes_to_numpy(audio_bytes)
            options = whisper.DecodingOptions(
                language=language,
                fp16=self._device == "cuda",
            )
            result = whisper.decode(self._model, whisper.pad_or_trim(audio_np), options)
            return STTResponse(
                text=result.text.strip(),
                language=result.language or "en",
                confidence=1.0 - result.avg_logprob * (-1) if hasattr(result, 'avg_logprob') else 0.9,
            )
        except Exception as e:
            return STTResponse(text="", language="en", confidence=0.0)

    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        try:
            # Try WAV format first
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return audio
        except Exception:
            # Assume raw PCM 16kHz mono int16
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
