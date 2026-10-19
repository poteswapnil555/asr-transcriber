import sys
import types

import numpy as np
import pytest

from conftest import FakeTranscriber
from asr_transcriber import synthetic as syn
from asr_transcriber.formats import Segment
from asr_transcriber.pipeline import transcribe_file
from asr_transcriber.transcriber import DEFAULT_MODEL, FasterWhisperTranscriber, TranscriptionError


# ---- pipeline: normal ------------------------------------------------------------------------------
def test_normal_file_gives_segments_text_and_timing(tone_wav):
    t = FakeTranscriber()
    r = transcribe_file(tone_wav, t)
    assert r.status == "ok" and r.model_called and t.calls == 1
    assert [s.text for s in r.segments] == ["Hello world.", "This is a test."]
    assert r.text == "Hello world. This is a test." and r.duration == pytest.approx(2.0)
    assert r.real_time_factor is not None and r.real_time_factor >= 0


def test_the_model_receives_16k_mono_float_samples(tmp_path):
    f = syn.write_wav(tmp_path / "s.wav", syn.tone(440, 1.0, sr=44100), sr=44100, channels=2)
    t = FakeTranscriber()
    transcribe_file(f, t)
    assert t.last_samples.dtype == np.float32 and t.last_samples.ndim == 1 and abs(len(t.last_samples) - 16000) <= 1


def test_mp3_input_works(tmp_path):
    try:
        f = syn.write_mp3(tmp_path / "t.mp3", syn.tone(440, 2.0))
    except Exception as e:
        pytest.skip(f"cannot encode mp3 here: {e}")
    assert transcribe_file(f, FakeTranscriber()).status == "ok"


def test_segments_from_the_model_are_sanitised(tone_wav):
    t = FakeTranscriber([Segment(3.0, 2.0, " late "), Segment(0.0, 1.0, "  "), Segment(float("nan"), 5.0, "nan start")])
    r = transcribe_file(tone_wav, t)
    assert [s.text for s in r.segments] == ["late", "nan start"] and all(s.end >= s.start >= 0 for s in r.segments)


# ---- pipeline: bad input never reaches the model and never crashes -------------------------------------
@pytest.mark.parametrize("make,fragment", [
    (lambda p: p / "nope.wav", "file not found"),
    (lambda p: p, "not a file"),
])
def test_missing_file_or_directory(tmp_path, make, fragment):
    t = FakeTranscriber()
    r = transcribe_file(make(tmp_path), t)
    assert r.status == "error" and fragment in r.message and t.calls == 0 and not r.model_called


def test_empty_file(tmp_path):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    t = FakeTranscriber()
    r = transcribe_file(f, t)
    assert r.status == "error" and "empty" in r.message and t.calls == 0


def test_corrupt_audio(tmp_path):
    f = tmp_path / "bad.wav"
    f.write_bytes(b"not audio" * 100)
    t = FakeTranscriber()
    r = transcribe_file(f, t)
    assert r.status == "error" and "not a valid WAV" in r.message and t.calls == 0


def test_unsupported_format(tmp_path):
    f = tmp_path / "talk.aac"
    f.write_bytes(b"data")
    r = transcribe_file(f, FakeTranscriber())
    assert r.status == "error" and "unsupported format" in r.message and ".wav" in r.message


def test_very_long_audio_is_refused_with_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_MINUTES", "0.02")
    f = syn.write_wav(tmp_path / "long.wav", syn.tone(440, 3.0))
    t = FakeTranscriber()
    r = transcribe_file(f, t)
    assert r.status == "error" and "limit is 0.02 minutes" in r.message and t.calls == 0


def test_bad_environment_value_is_a_clear_message_not_a_crash(tone_wav, monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_MINUTES", "lots")
    r = transcribe_file(tone_wav, FakeTranscriber())
    assert r.status == "error" and "MAX_AUDIO_MINUTES must be a number" in r.message


def test_silence_is_reported_without_running_the_model(silent_wav):
    t = FakeTranscriber()
    r = transcribe_file(silent_wav, t)
    assert r.status == "no_speech" and "silent" in r.message and t.calls == 0 and not r.model_called
    assert r.segments == [] and r.real_time_factor is None


def test_a_tone_passes_the_silence_check_and_the_model_decides(tone_wav):
    t = FakeTranscriber(segments=[])
    r = transcribe_file(tone_wav, t)
    assert r.status == "no_speech" and "model returned no text" in r.message and t.calls == 1 and r.model_called


def test_noise_only_audio_with_no_text_from_the_model(tmp_path):
    f = syn.write_wav(tmp_path / "n.wav", syn.noise(2.0, amplitude=0.05))
    assert transcribe_file(f, FakeTranscriber(segments=[Segment(0, 1, "   ")])).status == "no_speech"


# ---- pipeline: model failures ---------------------------------------------------------------------------
def test_missing_model_is_a_clear_error(tone_wav):
    r = transcribe_file(tone_wav, FakeTranscriber(error=TranscriptionError("could not load the model 'base.en'")))
    assert r.status == "error" and "could not load the model" in r.message and r.segments == []


def test_unexpected_model_exception_is_reported_not_raised(tone_wav):
    r = transcribe_file(tone_wav, FakeTranscriber(error=RuntimeError("cuda exploded")))
    assert r.status == "error" and "RuntimeError" in r.message and "cuda exploded" in r.message


def test_keyboard_interrupt_is_not_swallowed(tone_wav):
    with pytest.raises(KeyboardInterrupt):
        transcribe_file(tone_wav, FakeTranscriber(error=KeyboardInterrupt()))


# ---- FasterWhisperTranscriber with a fake faster_whisper module --------------------------------------------
class FakeSegment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def install_fake(monkeypatch, segments=None, init_error=None, transcribe_error=None):
    calls = {"init": None, "transcribe": None}

    class WhisperModel:
        def __init__(self, name, **kw):
            calls["init"] = (name, kw)
            if init_error:
                raise init_error

        def transcribe(self, samples, **kw):
            calls["transcribe"] = (samples, kw)
            if transcribe_error:
                raise transcribe_error
            return iter(segments or []), types.SimpleNamespace(language="en")

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return calls


def test_default_model_is_base_en():
    assert DEFAULT_MODEL == "base.en" and FasterWhisperTranscriber().name == "base.en"


def test_model_can_be_chosen_by_argument_or_environment(monkeypatch):
    assert FasterWhisperTranscriber("tiny.en").name == "tiny.en"
    monkeypatch.setenv("ASR_MODEL", "tiny.en")
    assert FasterWhisperTranscriber().name == "tiny.en"


def test_constructing_does_not_load_or_download_anything(monkeypatch):
    calls = install_fake(monkeypatch)
    FasterWhisperTranscriber("tiny.en")
    assert calls["init"] is None  # nothing happens until the first transcription


def test_transcribe_passes_expected_options_and_converts_segments(monkeypatch):
    calls = install_fake(monkeypatch, segments=[FakeSegment(0.0, 1.5, " Hello. "), FakeSegment(1.5, 2.0, "  "), FakeSegment(2.0, 3.0, "World.")])
    monkeypatch.delenv("ASR_OFFLINE", raising=False)
    monkeypatch.delenv("ASR_MODEL_DIR", raising=False)
    t = FasterWhisperTranscriber("tiny.en")
    out = t.transcribe(np.zeros(16000, dtype=np.float32))
    assert out == [Segment(0.0, 1.5, "Hello."), Segment(2.0, 3.0, "World.")]
    name, kw = calls["init"]
    assert name == "tiny.en" and kw["device"] == "cpu" and kw["compute_type"] == "int8" and kw["local_files_only"] is False
    samples, opts = calls["transcribe"]
    assert samples.dtype == np.float32 and opts["language"] == "en" and opts["vad_filter"] is True
    t.transcribe(np.zeros(16000, dtype=np.float32))  # second call reuses the loaded model
    assert calls["init"][0] == "tiny.en"


def test_multilingual_model_name_leaves_language_to_the_model(monkeypatch):
    calls = install_fake(monkeypatch)
    FasterWhisperTranscriber("base").transcribe(np.zeros(16000, dtype=np.float32))
    assert calls["transcribe"][1]["language"] is None


def test_language_can_be_set_by_environment(monkeypatch):
    calls = install_fake(monkeypatch)
    monkeypatch.setenv("ASR_LANGUAGE", "de")
    FasterWhisperTranscriber("base").transcribe(np.zeros(16000, dtype=np.float32))
    assert calls["transcribe"][1]["language"] == "de"


def test_offline_flag_and_model_dir_are_passed_through(monkeypatch):
    calls = install_fake(monkeypatch)
    monkeypatch.setenv("ASR_OFFLINE", "1")
    monkeypatch.setenv("ASR_MODEL_DIR", "some/dir")
    FasterWhisperTranscriber("tiny.en").transcribe(np.zeros(16000, dtype=np.float32))
    kw = calls["init"][1]
    assert kw["local_files_only"] is True and kw["download_root"] == "some/dir"


def test_package_not_installed_has_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes "import faster_whisper" raise ImportError
    with pytest.raises(TranscriptionError, match="pip install -r requirements.txt"):
        FasterWhisperTranscriber().transcribe(np.zeros(16000, dtype=np.float32))


def test_model_download_failure_has_a_clear_message_and_hint(monkeypatch):
    install_fake(monkeypatch, init_error=OSError("Cannot reach huggingface.co"))
    monkeypatch.delenv("ASR_OFFLINE", raising=False)
    with pytest.raises(TranscriptionError) as e:
        FasterWhisperTranscriber("base.en").transcribe(np.zeros(16000, dtype=np.float32))
    msg = str(e.value)
    assert "could not load the model 'base.en'" in msg and "OSError" in msg and "ASR_MODEL_DIR" in msg


def test_offline_with_missing_model_says_so(monkeypatch):
    install_fake(monkeypatch, init_error=OSError("not in cache"))
    monkeypatch.setenv("ASR_OFFLINE", "1")
    with pytest.raises(TranscriptionError, match="ASR_OFFLINE is set and the model is not on disk"):
        FasterWhisperTranscriber().transcribe(np.zeros(16000, dtype=np.float32))


def test_failure_during_transcription_is_wrapped(monkeypatch):
    install_fake(monkeypatch, transcribe_error=RuntimeError("decoder failed"))
    with pytest.raises(TranscriptionError, match="transcription failed.*decoder failed"):
        FasterWhisperTranscriber().transcribe(np.zeros(16000, dtype=np.float32))


def test_a_failed_load_is_retried_on_the_next_call(monkeypatch):
    install_fake(monkeypatch, init_error=OSError("offline"))
    t = FasterWhisperTranscriber()
    for _ in range(2):
        with pytest.raises(TranscriptionError):
            t.transcribe(np.zeros(16000, dtype=np.float32))


def test_bad_audio_with_the_real_class_never_touches_the_model(tmp_path, monkeypatch):
    calls = install_fake(monkeypatch)
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    r = transcribe_file(f, FasterWhisperTranscriber())
    assert r.status == "error" and calls["init"] is None  # no model was loaded or downloaded for a bad file


# ---- loading is separate from (and not counted in) transcription -------------------------------------------
from conftest import LoadingFake  # noqa: E402


def test_pipeline_loads_the_model_before_transcribing_and_does_not_time_the_load(tone_wav):
    t = LoadingFake(load_sleep=0.3)
    r = transcribe_file(tone_wav, t)
    assert t.events == ["load", "transcribe"] and r.status == "ok"
    assert r.seconds_taken < 0.2  # the 0.3 s load is not counted as transcription time


def test_pipeline_does_not_load_the_model_for_bad_or_silent_files(tmp_path, silent_wav):
    t = LoadingFake()
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert transcribe_file(empty, t).status == "error" and transcribe_file(silent_wav, t).status == "no_speech"
    assert t.loads == 0 and t.events == []


def test_pipeline_reports_a_load_failure_as_an_error_result(tone_wav):
    r = transcribe_file(tone_wav, LoadingFake(load_error=TranscriptionError("could not load the model 'base.en' (OSError: offline)")))
    assert r.status == "error" and "could not load the model" in r.message and not r.model_called


def test_pipeline_reports_an_unexpected_load_exception(tone_wav):
    r = transcribe_file(tone_wav, LoadingFake(load_error=MemoryError("out of memory")))
    assert r.status == "error" and "could not load the model (MemoryError" in r.message


def test_real_class_load_is_done_once_and_transcribe_reuses_it(monkeypatch):
    calls = install_fake(monkeypatch, segments=[FakeSegment(0, 1, "hi")])
    t = FasterWhisperTranscriber("tiny.en")
    t.load()
    assert calls["init"][0] == "tiny.en" and calls["transcribe"] is None
    first = calls["init"]
    t.transcribe(np.zeros(16000, dtype=np.float32))
    assert calls["init"] is first  # the model was not created again
