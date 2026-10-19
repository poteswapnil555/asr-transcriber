import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "LibriSpeech/test-clean"


def _script():
    spec = importlib.util.spec_from_file_location("download_clips", ROOT / "scripts" / "download_clips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_archive(chapters, trans_first=False, extra=(), gz=True, payload=None):
    """chapters: {(speaker, chapter): [(utterance_number, text), ...]}. Returns tar.gz bytes in LibriSpeech layout."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if gz else "w") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        for (spk, chap), utts in chapters.items():
            trans = "".join(f"{spk}-{chap}-{n:04d} {text}\n" for n, text in utts).encode()
            if trans_first:
                add(f"{PREFIX}/{spk}/{chap}/{spk}-{chap}.trans.txt", trans)
            for n, _ in utts:
                add(f"{PREFIX}/{spk}/{chap}/{spk}-{chap}-{n:04d}.flac", payload(n) if payload else f"FLAC-{spk}-{chap}-{n}".encode())
            if not trans_first:
                add(f"{PREFIX}/{spk}/{chap}/{spk}-{chap}.trans.txt", trans)
        for name, data in extra:
            add(name, data)
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(monkeypatch, mod, payloads):
    """payloads: list of bytes or Exceptions, one per mirror attempt."""
    calls = []

    def fake(req, timeout=0):
        calls.append(req.full_url)
        p = payloads[min(len(calls) - 1, len(payloads) - 1)]
        if isinstance(p, Exception):
            raise p
        return _Resp(p)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
    return calls


CHAPTER = {(1089, 134686): [(i, f"TEXT NUMBER {i} OF THE FIRST CHAPTER") for i in range(14)]}


@pytest.mark.parametrize("trans_first", [False, True])
def test_takes_the_first_n_clips_with_their_references(tmp_path, monkeypatch, trans_first):
    mod = _script()
    patch_urlopen(monkeypatch, mod, [make_archive(CHAPTER, trans_first)])
    assert mod.download(tmp_path / "clips", n=10) == 0
    manifest = json.loads((tmp_path / "clips" / "manifest.json").read_text())
    assert [c["id"] for c in manifest] == [f"1089-134686-{i:04d}" for i in range(10)]
    assert manifest[3] == {"id": "1089-134686-0003", "file": "1089-134686-0003.flac", "reference": "TEXT NUMBER 3 OF THE FIRST CHAPTER"}
    assert (tmp_path / "clips" / "1089-134686-0009.flac").read_bytes() == b"FLAC-1089-134686-9"
    assert not (tmp_path / "clips" / "1089-134686-0010.flac").exists()


def test_clips_can_come_from_several_chapters(tmp_path, monkeypatch):
    mod = _script()
    chapters = {(1, 2): [(0, "A ONE"), (1, "A TWO")], (3, 4): [(0, "B ONE"), (1, "B TWO")]}
    patch_urlopen(monkeypatch, mod, [make_archive(chapters)])
    assert mod.download(tmp_path, n=3) == 0
    assert [c["id"] for c in json.loads((tmp_path / "manifest.json").read_text())] == ["1-2-0000", "1-2-0001", "3-4-0000"]


def test_fewer_clips_than_asked_is_accepted_with_a_note(tmp_path, monkeypatch, capsys):
    mod = _script()
    patch_urlopen(monkeypatch, mod, [make_archive({(1, 2): [(0, "ONLY ONE"), (1, "ONLY TWO")]})])
    assert mod.download(tmp_path, n=10) == 0
    assert len(json.loads((tmp_path / "manifest.json").read_text())) == 2 and "only 2 clips" in capsys.readouterr().err


def test_it_stops_reading_as_soon_as_enough_clips_are_found(tmp_path, monkeypatch):
    mod = _script()
    many = {(1, 2): [(i, f"T {i}") for i in range(30)]}
    import random

    data = make_archive(many, trans_first=True, payload=lambda n: random.Random(n).randbytes(40_000))  # incompressible
    holder = {}

    class Counting(_Resp):
        def read(self, n=-1):
            out = super().read(n)
            holder["bytes"] = holder.get("bytes", 0) + len(out)
            return out

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: Counting(data))
    assert mod.download(tmp_path, n=3) == 0
    assert len(data) > 1_000_000 and holder["bytes"] < len(data) // 3  # most of the archive was never read


def test_unrelated_and_dangerous_member_names_are_ignored(tmp_path, monkeypatch):
    mod = _script()
    extra = [("../../evil.flac", b"x"), ("/etc/passwd", b"x"), (f"{PREFIX}/1/2/../../../escape.flac", b"x"),
             (f"{PREFIX}/1/2/1-2-0000.txt", b"x"), ("LibriSpeech/dev-clean/9/9/9-9-0000.flac", b"x"), ("README.md", b"x")]
    patch_urlopen(monkeypatch, mod, [make_archive({(1, 2): [(0, "GOOD ONE")]}, extra=extra)])
    assert mod.download(tmp_path / "clips", n=5) == 0
    assert sorted(p.name for p in (tmp_path / "clips").iterdir()) == ["1-2-0000.flac", "manifest.json"]
    assert not (tmp_path / "evil.flac").exists() and not (tmp_path.parent / "evil.flac").exists()


def test_oversized_clip_is_skipped(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod, "MAX_CLIP_BYTES", 5)
    patch_urlopen(monkeypatch, mod, [make_archive({(1, 2): [(0, "A")]})])
    assert mod.download(tmp_path, n=1) == 1  # the only clip was too large, so nothing usable was found


def test_a_transcript_line_for_a_missing_clip_is_ignored(tmp_path, monkeypatch):
    mod = _script()
    data = make_archive({(1, 2): [(0, "HAS AUDIO")]}, extra=[(f"{PREFIX}/1/2/1-2.trans.txt", b"1-2-0000 HAS AUDIO\n1-2-0005 NO AUDIO\n")])
    patch_urlopen(monkeypatch, mod, [data])
    assert mod.download(tmp_path, n=5) == 0
    assert [c["id"] for c in json.loads((tmp_path / "manifest.json").read_text())] == ["1-2-0000"]


# ---- failures ---------------------------------------------------------------------------------------------
def test_falls_back_to_the_next_mirror(tmp_path, monkeypatch):
    mod = _script()
    calls = patch_urlopen(monkeypatch, mod, [mod.urllib.error.URLError("mirror down"), make_archive(CHAPTER)])
    assert mod.download(tmp_path, n=2) == 0 and len(calls) == 2 and calls[0] != calls[1]


def test_all_mirrors_down_is_a_clear_message(tmp_path, monkeypatch, capsys):
    mod = _script()
    patch_urlopen(monkeypatch, mod, [mod.urllib.error.URLError("offline")])
    assert mod.download(tmp_path, n=2) == 1
    assert "Download failed" in capsys.readouterr().err and not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("payload", [b"<html>not an archive</html>", b"", b"\x1f\x8b" + b"broken" * 20])
def test_not_a_valid_archive_is_a_clear_message(tmp_path, monkeypatch, capsys, payload):
    mod = _script()
    patch_urlopen(monkeypatch, mod, [payload])
    assert mod.download(tmp_path, n=2) == 1
    assert "corrupt" in capsys.readouterr().err and not (tmp_path / "manifest.json").exists()


def test_a_truncated_archive_before_enough_clips_is_a_clean_failure_or_partial(tmp_path, monkeypatch):
    mod = _script()
    data = make_archive(CHAPTER)
    patch_urlopen(monkeypatch, mod, [data[: len(data) // 2]])
    assert mod.download(tmp_path, n=10) in (0, 1)  # never an exception


def test_archive_without_any_clip_is_reported(tmp_path, monkeypatch, capsys):
    mod = _script()
    patch_urlopen(monkeypatch, mod, [make_archive({}, extra=[("README.md", b"hello")])])
    assert mod.download(tmp_path, n=2) == 1 and "no clips with transcripts" in capsys.readouterr().err


def test_runaway_download_is_capped(tmp_path, monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod, "MAX_READ_BYTES", 200)
    patch_urlopen(monkeypatch, mod, [make_archive({(1, 2): [(i, "T") for i in range(50)]})])
    assert mod.download(tmp_path, n=50) == 1 and "stopped: read more than" in capsys.readouterr().err


def test_existing_manifest_is_kept_unless_forced(tmp_path, monkeypatch):
    mod = _script()
    (tmp_path / "manifest.json").write_text("[]")
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    assert mod.download(tmp_path, n=2) == 0 and (tmp_path / "manifest.json").read_text() == "[]"


def test_bad_n_is_a_usage_message(capsys):
    assert _script().main(["--n", "0"]) == 2 and "--n must be" in capsys.readouterr().err
