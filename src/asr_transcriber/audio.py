"""Load an audio file as 16 kHz mono float32 samples. Every failure is an AudioError with a clear message.

- .wav is decoded with the standard library (8/16/24/32-bit PCM); other sample rates use a plain linear resampler,
  so for best results give 16 kHz audio
- .mp3 .flac .ogg .m4a are decoded with PyAV (installed together with faster-whisper), which also resamples
- empty, corrupt, unsupported, too-short and too-long audio are refused here, before any model is touched
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import env_float

TARGET_SR = 16000
SUPPORTED = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
MIN_SECONDS = 0.1


class AudioError(Exception):
    """The audio cannot be used. The message is safe to show to the user."""


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32, mono, 16 kHz, range -1..1
    duration: float  # seconds

    @property
    def rms_dbfs(self) -> float:
        return rms_dbfs(self.samples)


def max_minutes() -> float:
    return env_float("MAX_AUDIO_MINUTES", 30.0)


def silence_dbfs() -> float:
    return env_float("SILENCE_DBFS", -50.0)


def rms_dbfs(samples: np.ndarray) -> float:
    """Loudness of the signal in dB relative to full scale. Digital silence is about -180."""
    if samples.size == 0:
        return -180.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return 20.0 * float(np.log10(max(rms, 1e-9)))


def is_silent(samples: np.ndarray, threshold_dbfs: float | None = None) -> bool:
    return rms_dbfs(samples) < (silence_dbfs() if threshold_dbfs is None else threshold_dbfs)


def _check_length(seconds: float, name: str) -> None:
    limit = max_minutes()
    if seconds > limit * 60:
        raise AudioError(f"{name}: the audio is {seconds / 60:.1f} minutes long; the limit is {limit:g} minutes "
                         "(set MAX_AUDIO_MINUTES to change it, or split the file)")
    if seconds < MIN_SECONDS:
        raise AudioError(f"{name}: the audio is too short ({seconds:.2f} s) to contain speech")


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_SR:
        return samples
    n_out = max(int(round(len(samples) * TARGET_SR / rate)), 1)
    return np.interp(np.linspace(0, len(samples) - 1, n_out), np.arange(len(samples)), samples).astype(np.float32)


def _decode_pcm(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 1:  # unsigned 8-bit
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        data = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise AudioError(f"unsupported WAV sample width ({width * 8}-bit)")
    if channels > 1:
        data = data[: len(data) // channels * channels].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32)


def _load_wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if rate <= 0 or channels <= 0:
                raise AudioError(f"{path.name}: the WAV header is invalid")
            _check_length(frames / rate, path.name)  # refuse long files before reading them into memory
            raw = w.readframes(frames)
    except AudioError:
        raise
    except (wave.Error, EOFError, OSError, ValueError) as e:
        raise AudioError(f"{path.name}: not a valid WAV file ({type(e).__name__}: {e})") from None
    return _resample(_decode_pcm(raw, width, channels), rate)


def _load_with_av(path: Path) -> np.ndarray:
    try:
        import av
    except ImportError:
        raise AudioError(f"{path.name}: decoding {path.suffix} needs PyAV, which comes with faster-whisper. "
                         "Run: pip install -r requirements.txt (or convert the file to .wav)") from None
    limit_samples = int(max_minutes() * 60 * TARGET_SR)
    try:
        chunks, total = [], 0
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioError(f"{path.name}: the file has no audio stream")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    arr = out.to_ndarray().reshape(-1)
                    chunks.append(arr)
                    total += arr.size
                if total > limit_samples:
                    _check_length(total / TARGET_SR, path.name)  # raises: too long (stop decoding early)
            for out in resampler.resample(None) or []:
                chunks.append(out.to_ndarray().reshape(-1))
    except AudioError:
        raise
    except Exception as e:  # PyAV raises many error types for corrupt input
        raise AudioError(f"{path.name}: could not decode the audio, the file may be corrupt ({type(e).__name__}: {str(e)[:120]})") from None
    if not chunks:
        raise AudioError(f"{path.name}: the file contains no audio samples")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def load_audio(path: Path | str) -> Audio:
    path = Path(path)
    if not path.exists():
        raise AudioError(f"file not found: {path}")
    if not path.is_file():
        raise AudioError(f"not a file: {path}")
    try:
        size = path.stat().st_size
    except OSError as e:
        raise AudioError(f"cannot read {path.name}: {e}") from None
    if size == 0:
        raise AudioError(f"{path.name}: the file is empty")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise AudioError(f"{path.name}: unsupported format '{path.suffix or '(none)'}'. Supported: {', '.join(SUPPORTED)}")

    samples = _load_wav(path) if suffix == ".wav" else _load_with_av(path)
    if samples.size == 0:
        raise AudioError(f"{path.name}: the file contains no audio samples")
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    duration = samples.size / TARGET_SR
    _check_length(duration, path.name)
    return Audio(np.clip(samples, -1.0, 1.0), duration)
