"""Voice Activity Detector using WebRTC VAD."""
from __future__ import annotations

import logging
import struct
from collections import deque

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Sliding window VAD using WebRTC VAD library.
    Returns True when a speech segment has ended (silence detected after speech)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        speech_buffer_size: int = 10,
        silence_threshold: int = 5,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # bytes

        self._speech_buffer: deque[bool] = deque(maxlen=speech_buffer_size)
        self._silence_counter = 0
        self._silence_threshold = silence_threshold
        self._in_speech = False

        self._vad = None
        self._init_vad(aggressiveness)

    def _init_vad(self, aggressiveness: int) -> None:
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
        except ImportError:
            logger.warning("webrtcvad not installed — using energy-based VAD fallback")

    def process_chunk(self, audio_chunk: bytes) -> bool:
        """Process one audio chunk. Returns True if speech just ended."""
        if len(audio_chunk) < self._frame_size:
            return False

        is_speech = self._detect_speech(audio_chunk[: self._frame_size])
        self._speech_buffer.append(is_speech)

        if is_speech:
            self._in_speech = True
            self._silence_counter = 0
        elif self._in_speech:
            self._silence_counter += 1
            if self._silence_counter >= self._silence_threshold:
                self._in_speech = False
                self._silence_counter = 0
                return True

        return False

    def _detect_speech(self, frame: bytes) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self._sample_rate)
            except Exception:
                pass

        # Energy-based fallback
        samples = struct.unpack(f"<{len(frame) // 2}h", frame[: (len(frame) // 2) * 2])
        if not samples:
            return False
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > 300  # empirical threshold for 16-bit PCM

    def reset(self) -> None:
        self._speech_buffer.clear()
        self._silence_counter = 0
        self._in_speech = False
