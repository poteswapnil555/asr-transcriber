import builtins

import numpy as np
import pytest

from asr_transcriber import synthetic as syn
from asr_transcriber.config import ConfigError
from asr_transcriber.audio import MIN_SECONDS, SUPPORTED, TARGET_SR, AudioError, is_silent, load_audio, rms_dbfs


def test_mono_16k_wav_loads_unchanged_in_length(tmp_path):
    f = syn.write_wav(tmp_path / "a.wav", syn.tone(440, 2.0))
    a = load_audio(f)
    assert a.samples.dtype == np.float32 and a.samples.ndim == 1
    assert len(a.samples) == 2 * TARGET_SR and a.duration == pytest.approx(2.0)
    assert -1.0 <= a.samples.min() and a.samples.max() <= 1.0 and a.samples.max() > 0.25


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_all_pcm_widths_decode_to_the_same_signal(tmp_path, width):
    original = syn.tone(300, 1.0, amplitude=0.5)
    a = load_audio(syn.write_wav(tmp_path / f"w{width}.wav", original, width=width))
    assert len(a.samples) == len(original)
    assert np.abs(a.samples - original).max() < (0.02 if width == 1 else 0.001)


def test_stereo_is_mixed_down_to_mono(tmp_path):
    a = load_audio(syn.write_wav(tmp_path / "s.wav", syn.tone(440, 1.0), channels=2))
    assert a.samples.ndim == 1 and len(a.samples) == TARGET_SR


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_other_sample_rates_are_resampled_to_16k(tmp_path, rate):
    a = load_audio(syn.write_wav(tmp_path / "r.wav", syn.tone(440, 1.0, sr=rate), sr=rate))
    assert abs(len(a.samples) - TARGET_SR) <= 1 and a.duration == pytest.approx(1.0, abs=0.01)


# ---- bad files ----------------------------------------------------------------------------
def test_missing_file(tmp_path):
    with pytest.raises(AudioError, match="file not found"):
        load_audio(tmp_path / "nope.wav")


def test_directory_is_not_a_file(tmp_path):
    with pytest.raises(AudioError, match="not a file"):
        load_audio(tmp_path)


@pytest.mark.parametrize("name", ["empty.wav", "empty.mp3", "empty.flac"])
def test_empty_file(tmp_path, name):
    f = tmp_path / name
    f.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        load_audio(f)


@pytest.mark.parametrize("name", ["notes.txt", "clip.aac", "clip.wma", "clip.mp4", "noextension", "audio.WAVE"])
def test_unsupported_format_lists_the_supported_ones(tmp_path, name):
    f = tmp_path / name
    f.write_bytes(b"data")
    with pytest.raises(AudioError, match="unsupported format") as e:
        load_audio(f)
    assert all(s in str(e.value) for s in SUPPORTED)


def test_extension_is_case_insensitive(tmp_path):
    assert load_audio(syn.write_wav(tmp_path / "LOUD.WAV", syn.tone(440, 1.0))).duration == pytest.approx(1.0)


@pytest.mark.parametrize("payload", [b"this is not audio at all" * 50, b"RIFF" + b"\x00" * 100, bytes(range(256)) * 10])
def test_corrupt_wav(tmp_path, payload):
    f = tmp_path / "bad.wav"
    f.write_bytes(payload)
    with pytest.raises(AudioError, match="not a valid WAV"):
        load_audio(f)


def test_wav_with_header_only_has_no_samples(tmp_path):
    f = syn.write_wav(tmp_path / "h.wav", np.zeros(0, dtype=np.float32))
    with pytest.raises(AudioError, match="too short|no audio samples"):
        load_audio(f)


def test_very_short_audio_is_refused(tmp_path):
    f = syn.write_wav(tmp_path / "short.wav", syn.tone(440, MIN_SECONDS / 2))
    with pytest.raises(AudioError, match="too short"):
        load_audio(f)


def test_very_long_audio_is_refused_with_the_limit_in_the_message(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_MINUTES", "0.02")  # 1.2 seconds
    f = syn.write_wav(tmp_path / "long.wav", syn.silence(3.0))
    with pytest.raises(AudioError, match="MAX_AUDIO_MINUTES") as e:
        load_audio(f)
    assert "limit is 0.02 minutes" in str(e.value)


def test_a_long_wav_is_refused_from_its_header_without_decoding(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_MINUTES", "0.02")
    f = syn.write_wav(tmp_path / "long.wav", syn.silence(3.0))
    import asr_transcriber.audio as audio

    monkeypatch.setattr(audio, "_decode_pcm", lambda *a: pytest.fail("must not decode"))
    with pytest.raises(AudioError, match="minutes long"):
        load_audio(f)


def test_bad_limit_value_is_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_MINUTES", "lots")
    f = syn.write_wav(tmp_path / "a.wav", syn.tone(440, 1.0))
    with pytest.raises(ConfigError, match="MAX_AUDIO_MINUTES must be a number"):
        load_audio(f)


# ---- silence --------------------------------------------------------------------------------
def test_digital_silence_is_silent_and_a_tone_is_not():
    assert is_silent(syn.silence(1.0)) and rms_dbfs(syn.silence(1.0)) < -100
    assert not is_silent(syn.tone(440, 1.0))
    assert rms_dbfs(syn.tone(440, 1.0, amplitude=0.5)) == pytest.approx(-9.0, abs=0.1)  # 0.5/sqrt(2)


def test_very_quiet_noise_is_silent_but_louder_noise_is_not():
    assert is_silent(syn.noise(1.0, amplitude=0.0005))
    assert not is_silent(syn.noise(1.0, amplitude=0.05))


def test_silence_threshold_is_configurable(monkeypatch):
    quiet = syn.noise(1.0, amplitude=0.005)  # about -46 dBFS
    assert not is_silent(quiet)
    monkeypatch.setenv("SILENCE_DBFS", "-30")
    assert is_silent(quiet)


def test_empty_samples_count_as_silent():
    assert is_silent(np.zeros(0, dtype=np.float32))


# ---- other formats via PyAV ----------------------------------------------------------------------
def test_real_mp3_is_decoded(tmp_path):
    try:
        f = syn.write_mp3(tmp_path / "t.mp3", syn.tone(440, 2.0))
    except Exception as e:  # encoder missing in this PyAV build
        pytest.skip(f"cannot encode mp3 here: {e}")
    a = load_audio(f)
    assert a.duration == pytest.approx(2.0, abs=0.2) and not is_silent(a.samples)


def test_corrupt_mp3_gives_a_clear_message(tmp_path):
    f = tmp_path / "bad.mp3"
    f.write_bytes(b"\xff\xfb\x90\x00" + b"not really mp3 data" * 100)
    with pytest.raises(AudioError, match="could not decode|no audio"):
        load_audio(f)


def test_garbage_flac_gives_a_clear_message(tmp_path):
    f = tmp_path / "bad.flac"
    f.write_bytes(b"fLaC" + b"\x00" * 50)
    with pytest.raises(AudioError, match="could not decode|no audio"):
        load_audio(f)


def test_mp3_without_pyav_installed_says_what_to_install(tmp_path, monkeypatch):
    f = tmp_path / "x.mp3"
    f.write_bytes(b"data")
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "av":
            raise ImportError("no av")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AudioError, match="needs PyAV"):
        load_audio(f)


def test_a_wav_still_loads_without_pyav(tmp_path, monkeypatch):
    f = syn.write_wav(tmp_path / "a.wav", syn.tone(440, 1.0))
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "av" else real_import(name, *a, **k))
    assert load_audio(f).duration == pytest.approx(1.0)


def test_long_mp3_is_refused(tmp_path, monkeypatch):
    try:
        f = syn.write_mp3(tmp_path / "long.mp3", syn.silence(4.0))
    except Exception as e:
        pytest.skip(f"cannot encode mp3 here: {e}")
    monkeypatch.setenv("MAX_AUDIO_MINUTES", "0.02")
    with pytest.raises(AudioError, match="minutes long"):
        load_audio(f)
