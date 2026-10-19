"""Speech-to-text with a local faster-whisper model. No API key, no LLM.

The model is loaded on first use (not when the object is created), so bad input files are refused before anything
is downloaded. Whisper gives text and timestamps only; it does not identify speakers.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import ConfigError, env, env_bool
from .formats import Segment

DEFAULT_MODEL = "base.en"


class TranscriptionError(Exception):
    """The model could not be loaded or failed while transcribing. The message is safe to show to the user."""


class Transcriber(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        """samples: float32 mono 16 kHz. Returns timed segments (an empty list means no speech was found)."""
        ...


class FasterWhisperTranscriber:
    def __init__(self, model: str | None = None, device: str = "cpu", compute_type: str = "int8",
                 vad_filter: bool = True) -> None:
        self.name = model or env("ASR_MODEL", DEFAULT_MODEL)
        self.device, self.compute_type, self.vad_filter = device, compute_type, vad_filter
        self._model = None  # loaded on first use

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt") from None
        offline = env_bool("ASR_OFFLINE", False)
        try:
            self._model = WhisperModel(self.name, device=self.device, compute_type=self.compute_type,
                                       download_root=env("ASR_MODEL_DIR"), local_files_only=offline)
        except Exception as e:  # missing model, no network, bad model name, corrupt cache, out of memory...
            hint = ("ASR_OFFLINE is set and the model is not on disk; download it once without ASR_OFFLINE"
                    if offline else
                    "if you are offline, download the model once while online, or set ASR_MODEL_DIR to a folder that already has it")
            raise TranscriptionError(
                f"could not load the model '{self.name}' ({type(e).__name__}: {str(e)[:160]}). Hint: {hint}.") from None
        return self._model

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        model = self._load()
        language = env("ASR_LANGUAGE") or ("en" if self.name.endswith(".en") else None)
        try:
            raw_segments, _info = model.transcribe(
                samples, language=language, beam_size=5, vad_filter=self.vad_filter, condition_on_previous_text=False)
            return [Segment(float(s.start), float(s.end), str(s.text).strip()) for s in raw_segments if str(s.text).strip()]
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"transcription failed ({type(e).__name__}: {str(e)[:160]})") from None


__all__ = ["DEFAULT_MODEL", "ConfigError", "FasterWhisperTranscriber", "Transcriber", "TranscriptionError"]
