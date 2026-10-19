import json
from pathlib import Path

import pytest

from conftest import FakeTranscriber
from asr_transcriber import __main__ as cli
from asr_transcriber import evaluate as ev
from asr_transcriber import synthetic as syn
from asr_transcriber.formats import Segment
from asr_transcriber.transcriber import TranscriptionError

from test_formats import read_vtt  # the strict WebVTT reader used for the format contract


# ---- CLI --------------------------------------------------------------------------------------------
@pytest.fixture
def fake(monkeypatch):
    t = FakeTranscriber()
    monkeypatch.setattr(cli, "make_transcriber", lambda model: t)
    return t


def test_cli_writes_text_and_vtt_with_speaker_unknown(tone_wav, tmp_path, fake, capsys):
    out = tmp_path / "out"
    assert cli.main([str(tone_wav), "--out-dir", str(out)]) == 0
    assert (out / "tone.txt").read_text(encoding="utf-8") == "Hello world.\nThis is a test.\n"
    cues = read_vtt((out / "tone.vtt").read_text(encoding="utf-8"))
    assert [(c[3], c[4]) for c in cues] == [("Speaker ?", "Hello world."), ("Speaker ?", "This is a test.")]
    msg = capsys.readouterr().out
    assert msg.startswith("ok: 2 segments from 2.0 s of audio") and "tone.vtt" in msg


def test_cli_no_speaker_option(tone_wav, tmp_path, fake):
    assert cli.main([str(tone_wav), "--out-dir", str(tmp_path), "--no-speaker"]) == 0
    text = (tmp_path / "tone.vtt").read_text(encoding="utf-8")
    assert "<v" not in text and [c[3] for c in read_vtt(text)] == [None, None]


def test_cli_custom_speaker_label_and_print(tone_wav, tmp_path, fake, capsys):
    assert cli.main([str(tone_wav), "--out-dir", str(tmp_path), "--speaker-label", "Unknown", "--print"]) == 0
    assert read_vtt((tmp_path / "tone.vtt").read_text(encoding="utf-8"))[0][3] == "Unknown"
    assert "Hello world.\nThis is a test.\n" in capsys.readouterr().out


def test_cli_creates_the_output_folder_and_overwrites_existing_files(tone_wav, tmp_path, fake):
    out = tmp_path / "a" / "b"
    (out).mkdir(parents=True)
    (out / "tone.txt").write_text("old")
    assert cli.main([str(tone_wav), "--out-dir", str(out)]) == 0
    assert (out / "tone.txt").read_text(encoding="utf-8").startswith("Hello world.")


@pytest.mark.parametrize("name,write,fragment", [
    ("missing.wav", None, "file not found"),
    ("empty.wav", b"", "empty"),
    ("bad.wav", b"junk" * 100, "not a valid WAV"),
    ("notes.txt", b"hello", "unsupported format"),
])
def test_cli_bad_input_prints_a_clear_message_and_exits_1(tmp_path, fake, capsys, name, write, fragment):
    f = tmp_path / name
    if write is not None:
        f.write_bytes(write)
    assert cli.main([str(f), "--out-dir", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and fragment in err
    assert fake.calls == 0 and not (tmp_path / "o").exists()  # nothing written, model never called


def test_cli_silence_writes_nothing(silent_wav, tmp_path, fake, capsys):
    assert cli.main([str(silent_wav), "--out-dir", str(tmp_path / "o")]) == 1
    assert "no_speech" in capsys.readouterr().err and not (tmp_path / "o").exists() and fake.calls == 0


def test_cli_model_missing_gives_the_message_and_exit_1(tone_wav, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_transcriber",
                        lambda m: FakeTranscriber(error=TranscriptionError("could not load the model 'base.en' (OSError: offline). Hint: download it first.")))
    assert cli.main([str(tone_wav), "--out-dir", str(tmp_path / "o")]) == 1
    assert "could not load the model" in capsys.readouterr().err


def test_cli_unwritable_output_location_is_a_message_not_a_traceback(tone_wav, tmp_path, fake, capsys):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    assert cli.main([str(tone_wav), "--out-dir", str(blocker)]) == 1
    assert "could not write the output files" in capsys.readouterr().err


def test_cli_without_arguments_is_a_usage_error():
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2


def test_cli_passes_the_model_name_through(tone_wav, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "make_transcriber", lambda model: seen.append(model) or FakeTranscriber())
    cli.main([str(tone_wav), "--out-dir", str(tmp_path), "--model", "tiny.en"])
    assert seen == ["tiny.en"]


def test_a_real_run_with_a_bad_file_does_not_load_the_model(tmp_path, capsys):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    assert cli.main([str(f), "--out-dir", str(tmp_path / "o")]) == 1  # the real FasterWhisperTranscriber class is used
    assert "empty" in capsys.readouterr().err


# ---- mock evaluation -----------------------------------------------------------------------------------
def test_mock_run_prints_not_accuracy_and_exact_known_numbers(capsys):
    assert ev.main(["--mock"]) == 0
    out = capsys.readouterr().out
    assert "MODE: mock" in out and "NOT accuracy" in out
    s = json.loads(out[out.index("{"):])
    # scripted errors: 2 clips lose a word, 1 word swapped, 1 word added, 1 clip empty (8 words missed)
    assert (s["clips"], s["clips_failed_or_no_speech"], s["reference_words"]) == (10, 1, 70)
    assert (s["substitutions"], s["deletions"], s["insertions"]) == (1, 10, 1)
    assert s["corpus_wer"] == pytest.approx(12 / 70, abs=1e-4) and s["mean_clip_wer"] == pytest.approx(0.1658, abs=1e-3)
    assert s["real_time_factor"] is None  # no fake speed numbers in mock mode


def test_mock_details_in_the_out_file(tmp_path):
    out = tmp_path / "e.json"
    assert ev.main(["--mock", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    rows = {r["id"]: r for r in data["clips_detail"]}
    assert len(rows) == 10 and rows["m03"]["deletions"] == 1 and rows["m04"]["substitutions"] == 1
    assert rows["m07"]["wer"] == 0.0  # punctuation differences are normalised away
    assert rows["m10"]["status"] == "no_speech" and rows["m10"]["deletions"] == 8 and rows["m10"]["wer"] == 1.0
    assert rows["m01"]["transcribe_seconds"] is None


def test_mock_limit(capsys):
    assert ev.main(["--mock", "--limit", "3"]) == 0
    assert json.loads(capsys.readouterr().out.split("MODE: mock")[1].split("\n", 1)[1])["clips"] == 3


def test_mock_and_live_cannot_be_combined():
    with pytest.raises(SystemExit):
        ev.main(["--mock", "--live"])


# ---- live mode (with a fake model; the real model is never used in these tests) ------------------------------
def make_clips(tmp_path, refs):
    for i, (cid, ref) in enumerate(refs.items()):
        syn.write_wav(tmp_path / f"{cid}.wav", syn.tone(300 + 50 * i, 2.0))
    (tmp_path / "manifest.json").write_text(json.dumps([{"id": c, "file": f"{c}.wav", "reference": r} for c, r in refs.items()]))
    return tmp_path


def test_live_without_clips_points_to_the_download_script(tmp_path, capsys):
    assert ev.main(["--live", "--clips", str(tmp_path)]) == 2
    assert "download_clips.py" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["not json", "[]", '{"a": 1}', '[{"id": "x"}]'])
def test_live_with_a_bad_manifest(tmp_path, capsys, content):
    (tmp_path / "manifest.json").write_text(content)
    assert ev.main(["--live", "--clips", str(tmp_path)]) == 2
    assert "Cannot run evaluation" in capsys.readouterr().err


def test_live_scores_clips_with_normalisation_and_reports_timing(tmp_path, monkeypatch, capsys):
    clips = make_clips(tmp_path, {"c1": "HE HOPED THERE WOULD BE STEW", "c2": "IT'S A GOOD DAY"})
    fake = FakeTranscriber([Segment(0, 1, "He hoped there would be stew.")])
    monkeypatch.setattr(ev, "FasterWhisperTranscriber", lambda model: fake)
    assert ev.main(["--live", "--clips", str(clips)]) == 0
    out = capsys.readouterr().out
    assert "MODE: live" in out
    s = json.loads(out[out.index("{"):])
    # c1 is perfect after normalising case and the full stop; c2 gets a wrong text: 4 words vs 6 words
    assert s["clips"] == 2 and s["reference_words"] == 10 and s["corpus_wer"] is not None and s["real_time_factor"] is not None


def test_live_model_failure_counts_every_clip_as_failed_without_crashing(tmp_path, monkeypatch, capsys):
    clips = make_clips(tmp_path, {"c1": "one two three", "c2": "four five six"})
    broken = FakeTranscriber(error=TranscriptionError("could not load the model 'base.en' (OSError: offline)"))
    monkeypatch.setattr(ev, "FasterWhisperTranscriber", lambda model: broken)
    assert ev.main(["--live", "--clips", str(clips)]) == 0
    cap = capsys.readouterr()
    s = json.loads(cap.out[cap.out.index("{"):])
    assert s["clips_failed_or_no_speech"] == 2 and s["corpus_wer"] == 1.0 and s["deletions"] == 6
    assert "could not load the model" in cap.err


def test_live_missing_audio_file_for_a_clip_is_a_failed_clip(tmp_path, monkeypatch, capsys):
    clips = make_clips(tmp_path, {"c1": "one two three"})
    (tmp_path / "c1.wav").unlink()
    monkeypatch.setattr(ev, "FasterWhisperTranscriber", lambda model: FakeTranscriber())
    assert ev.main(["--live", "--clips", str(clips)]) == 0
    assert "file not found" in capsys.readouterr().err


# ---- live mode: model loading is timed separately -------------------------------------------------------------
from conftest import LoadingFake  # noqa: E402


def test_live_reports_model_load_time_separately_and_excludes_it_from_the_real_time_factor(tmp_path, monkeypatch, capsys):
    clips = make_clips(tmp_path, {"c1": "one two three", "c2": "four five six"})
    fake = LoadingFake([Segment(0, 1, "one two three")], load_sleep=0.4)
    monkeypatch.setattr(ev, "FasterWhisperTranscriber", lambda model: fake)
    assert ev.main(["--live", "--clips", str(clips)]) == 0
    out = capsys.readouterr().out
    s = json.loads(out[out.index("{"):])
    assert fake.loads >= 1 and fake.events[0] == "load"
    assert s["model_load_seconds"] >= 0.4 and s["real_time_factor"] < 0.2  # load time is not in the speed number


def test_live_model_that_cannot_be_loaded_stops_with_a_clear_message(tmp_path, monkeypatch, capsys):
    clips = make_clips(tmp_path, {"c1": "one two three"})
    fake = LoadingFake(load_error=TranscriptionError("could not load the model 'base.en' (OSError: offline). Hint: download it first."))
    monkeypatch.setattr(ev, "FasterWhisperTranscriber", lambda model: fake)
    assert ev.main(["--live", "--clips", str(clips)]) == 2
    err = capsys.readouterr().err
    assert "Cannot run evaluation" in err and "could not load the model" in err


def test_mock_has_no_model_load_time(capsys):
    assert ev.main(["--mock"]) == 0
    assert "model_load_seconds" not in capsys.readouterr().out
