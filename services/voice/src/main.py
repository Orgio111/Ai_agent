"""Voice Service: streaming Whisper STT + edge-tts TTS + VAD pipeline.
Target latency: ≤300ms from speech end to first TTS token."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .stt import WhisperSTT
from .tts import EdgeTTS
from .vad import VoiceActivityDetector
from .models import TTSRequest, STTResponse, VoiceConfig

stt_engine: WhisperSTT | None = None
tts_engine: EdgeTTS | None = None
vad: VoiceActivityDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global stt_engine, tts_engine, vad

    stt_engine = WhisperSTT(
        model_size=os.environ.get("WHISPER_MODEL", "base"),
        device=os.environ.get("WHISPER_DEVICE", "cpu"),
    )
    await stt_engine.load()

    tts_engine = EdgeTTS(
        default_voice=os.environ.get("TTS_VOICE", "en-US-AriaNeural"),
        default_rate=os.environ.get("TTS_RATE", "+0%"),
    )

    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        aggressiveness=2,
    )

    yield

    await stt_engine.unload()


app = FastAPI(title="JARVIS Voice Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket) -> None:
    """Full-duplex voice WebSocket.
    Client streams raw PCM audio bytes → server streams TTS audio bytes back.
    Protocol:
      inbound: {"action": "audio", "data": "<base64 PCM>"}
              {"action": "config", "voice": "...", "rate": "..."}
      outbound: {"type": "stt", "text": "...", "partial": bool}
               {"type": "tts", "audio": "<base64 MP3>", "done": bool}
    """
    await ws.accept()
    assert stt_engine is not None and tts_engine is not None and vad is not None

    audio_buffer = bytearray()
    config = VoiceConfig()

    try:
        while True:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
            action = msg.get("action", "")

            if action == "audio":
                import base64
                chunk = base64.b64decode(msg.get("data", ""))
                audio_buffer.extend(chunk)

                # VAD: detect speech end
                speech_ended = vad.process_chunk(chunk)
                if speech_ended and len(audio_buffer) > 1000:
                    audio_data = bytes(audio_buffer)
                    audio_buffer.clear()

                    # STT
                    stt_result = await stt_engine.transcribe(audio_data)
                    await ws.send_json({
                        "type": "stt",
                        "text": stt_result.text,
                        "language": stt_result.language,
                        "confidence": stt_result.confidence,
                        "partial": False,
                    })

                    # Stream TTS response
                    if stt_result.text:
                        async for tts_chunk in tts_engine.synthesize_stream(
                            stt_result.text, voice=config.voice, rate=config.rate
                        ):
                            import base64
                            await ws.send_json({
                                "type": "tts",
                                "audio": base64.b64encode(tts_chunk).decode(),
                                "done": False,
                            })
                        await ws.send_json({"type": "tts", "audio": "", "done": True})

            elif action == "config":
                config.voice = msg.get("voice", config.voice)
                config.rate = msg.get("rate", config.rate)
                await ws.send_json({"type": "config_ack", "voice": config.voice})

            elif action == "tts_only":
                text = msg.get("text", "")
                if text:
                    async for chunk in tts_engine.synthesize_stream(
                        text, voice=config.voice, rate=config.rate
                    ):
                        import base64
                        await ws.send_json({
                            "type": "tts",
                            "audio": base64.b64encode(chunk).decode(),
                            "done": False,
                        })
                    await ws.send_json({"type": "tts", "audio": "", "done": True})

            elif action == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        await ws.close(code=1001)


@app.post("/stt")
async def transcribe_audio(request: dict) -> STTResponse:
    """Single-shot audio transcription via HTTP."""
    assert stt_engine is not None
    import base64
    audio_b64 = request.get("audio_b64", "")
    if not audio_b64:
        raise HTTPException(400, "No audio data provided")
    audio_data = base64.b64decode(audio_b64)
    return await stt_engine.transcribe(audio_data)


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Stream TTS audio for a text input."""
    assert tts_engine is not None
    from fastapi.responses import StreamingResponse
    import base64

    async def audio_generator():
        async for chunk in tts_engine.synthesize_stream(req.text, voice=req.voice, rate=req.rate):
            yield chunk

    return StreamingResponse(audio_generator(), media_type="audio/mpeg")


@app.get("/voices")
async def list_voices() -> dict:
    return {
        "voices": [
            {"id": "en-US-AriaNeural", "language": "en-US", "gender": "female"},
            {"id": "en-US-GuyNeural", "language": "en-US", "gender": "male"},
            {"id": "en-GB-SoniaNeural", "language": "en-GB", "gender": "female"},
            {"id": "zh-CN-XiaoxiaoNeural", "language": "zh-CN", "gender": "female"},
            {"id": "ru-RU-SvetlanaNeural", "language": "ru-RU", "gender": "female"},
            {"id": "mn-MN-NominchuluNeural", "language": "mn-MN", "gender": "female"},
        ]
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "stt_loaded": stt_engine is not None and stt_engine.loaded,
        "tts_ready": tts_engine is not None,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.environ.get("VOICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICE_PORT", "8005")),
        log_level="info",
    )
