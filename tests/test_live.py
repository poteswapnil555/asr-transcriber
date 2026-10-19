"""Optional live test: skipped unless faster-whisper is installed, the model is already on disk and a clip was downloaded.

It never downloads anything. To enable it: pip install -r requirements.txt, run the CLI or evaluation once so the model
is cached, and run scripts/download_clips.py.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "clips" / "manifest.json"


def _model_available():
    try:
        from faster_whisper import WhisperModel

        WhisperModel("tiny.en", device="cpu", compute_type="int8", local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not (MANIFEST.exists() and _model_available()),
                                reason="needs faster-whisper, tiny.en already cached, and data/clips (no downloads are done by tests)")


def test_live_transcription_of_a_librispeech_clip_has_a_sane_word_error_rate(monkeypatch):
    from asr_transcriber.pipeline import transcribe_file
    from asr_transcriber.transcriber import FasterWhisperTranscriber
    from asr_transcriber.wer import wer

    monkeypatch.setenv("ASR_OFFLINE", "1")
    clip = json.loads(MANIFEST.read_text())[0]
    r = transcribe_file(MANIFEST.parent / clip["file"], FasterWhisperTranscriber("tiny.en"))
    assert r.status == "ok" and r.segments
    assert wer(clip["reference"], r.text) < 0.5  # loose on purpose: this checks the plumbing, not the accuracy
