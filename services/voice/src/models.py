from __future__ import annotations

from pydantic import BaseModel


class STTResponse(BaseModel):
    text: str
    language: str = "en"
    confidence: float = 1.0
    duration_ms: float = 0.0


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class VoiceConfig(BaseModel):
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    language: str = "en"
