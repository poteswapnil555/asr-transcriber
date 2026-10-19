"""File -> checks -> transcriber -> segments. Never raises: every failure becomes a Result with a clear message."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio import AudioError, is_silent, load_audio, rms_dbfs
from .config import ConfigError
from .formats import Segment, sanitise
from .transcriber import Transcriber, TranscriptionError


@dataclass
class Result:
    status: str  # ok | no_speech | error
    segments: list[Segment] = field(default_factory=list)
    message: str = ""
    duration: float = 0.0  # seconds of audio
    seconds_taken: float = 0.0  # time spent transcribing
    model_called: bool = False

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def real_time_factor(self) -> float | None:
        """Processing time divided by audio length; below 1 means faster than real time."""
        return self.seconds_taken / self.duration if self.duration and self.model_called else None


def transcribe_file(path: Path | str, transcriber: Transcriber) -> Result:
    try:
        audio = load_audio(path)
    except AudioError as e:
        return Result("error", message=str(e))
    except ConfigError as e:
        return Result("error", message=f"configuration error: {e}")
    except Exception as e:  # anything unexpected while reading the file
        return Result("error", message=f"could not read the audio ({type(e).__name__}: {str(e)[:120]})")

    if is_silent(audio.samples):
        return Result("no_speech", duration=audio.duration,
                      message=f"no speech detected: the audio is silent ({rms_dbfs(audio.samples):.0f} dBFS). The model was not run")

    start = time.perf_counter()
    try:
        segments = sanitise(transcriber.transcribe(audio.samples))
    except TranscriptionError as e:
        return Result("error", duration=audio.duration, message=str(e))
    except Exception as e:
        return Result("error", duration=audio.duration, message=f"transcription failed ({type(e).__name__}: {str(e)[:120]})")
    taken = time.perf_counter() - start
    if not segments:
        return Result("no_speech", duration=audio.duration, seconds_taken=taken, model_called=True,
                      message="no speech detected: the model returned no text")
    return Result("ok", segments, duration=audio.duration, seconds_taken=taken, model_called=True)
