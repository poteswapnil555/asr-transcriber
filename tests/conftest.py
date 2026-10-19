import numpy as np
import pytest

from asr_transcriber import synthetic as syn
from asr_transcriber.formats import Segment


class FakeTranscriber:
    """Returns the given segments (or raises the given exception); counts calls and remembers what it was given."""

    name = "fake"

    def __init__(self, segments=None, error: Exception | None = None):
        self.segments = segments if segments is not None else [Segment(0.0, 2.0, "Hello world."), Segment(2.0, 4.5, "This is a test.")]
        self.error = error
        self.calls = 0
        self.last_samples: np.ndarray | None = None

    def transcribe(self, samples):
        self.calls += 1
        self.last_samples = samples
        if self.error:
            raise self.error
        return self.segments


@pytest.fixture
def tone_wav(tmp_path):
    return syn.write_wav(tmp_path / "tone.wav", syn.tone(440, 2.0))


@pytest.fixture
def silent_wav(tmp_path):
    return syn.write_wav(tmp_path / "silence.wav", syn.silence(2.0))
