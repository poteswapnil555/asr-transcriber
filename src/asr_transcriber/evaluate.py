"""Word error rate (WER) on a small set of clips with reference transcripts.

  python -m asr_transcriber.evaluate --mock   # plumbing check: synthetic tones + scripted fake transcriber. NOT accuracy.
  python -m asr_transcriber.evaluate --live   # the real model on the LibriSpeech clips from scripts/download_clips.py

Both sides are normalised before comparing (see wer.py). A clip that fails or yields no text counts as a clip whose
every reference word was missed (all deletions), so failures lower the score instead of being hidden.
Corpus WER = total errors / total reference words. Timing (real-time factor) is only reported for --live.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import synthetic as syn
from .config import ConfigError
from .pipeline import Result, transcribe_file
from .transcriber import FasterWhisperTranscriber
from .wer import Errors, corpus_errors, count_errors

CLIPS_DIR = Path("data/clips")

# Mock data: invented sentences with scripted "model output" containing known errors (plumbing check only)
MOCK_CLIPS = [
    ("m01", "the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog"),
    ("m02", "she sells sea shells by the sea shore", "she sells sea shells by the sea shore"),
    ("m03", "please call me back tomorrow morning", "please call me back tomorrow"),  # 1 deletion
    ("m04", "we will meet at the station at noon", "we will meet at the stadium at noon"),  # 1 substitution
    ("m05", "the weather is lovely today", "the weather is really lovely today"),  # 1 insertion
    ("m06", "open the window and let in some air", "open the window and let in some air"),
    ("m07", "the meeting starts at nine o'clock", "the meeting starts at nine o'clock."),  # punctuation only
    ("m08", "bring your laptop and a charger", "bring your laptop and charger"),  # 1 deletion
    ("m09", "thank you all for coming today", "thank you all for coming today"),
    ("m10", "this sentence is only here as a placeholder", ""),  # model found nothing
]


@dataclass
class ClipRow:
    id: str
    reference: str
    hypothesis: str
    status: str
    errors: Errors
    duration: float
    seconds_taken: float | None
    message: str = ""

    def as_dict(self) -> dict:
        e = self.errors
        return {"id": self.id, "status": self.status, "reference": self.reference, "hypothesis": self.hypothesis,
                "wer": None if e.wer is None else round(e.wer, 4), "substitutions": e.substitutions,
                "deletions": e.deletions, "insertions": e.insertions, "reference_words": e.reference_words,
                "audio_seconds": round(self.duration, 2),
                "transcribe_seconds": None if self.seconds_taken is None else round(self.seconds_taken, 2),
                "message": self.message}


@dataclass
class Report:
    mode: str
    model: str
    rows: list[ClipRow] = field(default_factory=list)

    def add(self, clip_id: str, reference: str, result: Result, timing: bool) -> None:
        hyp = result.text if result.status == "ok" else ""
        self.rows.append(ClipRow(clip_id, reference, hyp, result.status, count_errors(reference, hyp), result.duration,
                                 result.seconds_taken if timing and result.model_called else None, result.message))

    def summary(self) -> dict:
        pooled = corpus_errors([r.errors for r in self.rows])
        rates = [r.errors.wer for r in self.rows if r.errors.wer is not None]
        timed = [r for r in self.rows if r.seconds_taken is not None]
        audio = sum(r.duration for r in timed)
        taken = sum(r.seconds_taken for r in timed)
        return {
            "mode": self.mode, "model": self.model, "clips": len(self.rows),
            "clips_failed_or_no_speech": sum(r.status != "ok" for r in self.rows),
            "reference_words": pooled.reference_words,
            "substitutions": pooled.substitutions, "deletions": pooled.deletions, "insertions": pooled.insertions,
            "corpus_wer": None if pooled.wer is None else round(pooled.wer, 4),
            "mean_clip_wer": round(sum(rates) / len(rates), 4) if rates else None,
            "real_time_factor": round(taken / audio, 3) if audio else None,
        }


class ScriptedTranscriber:
    """Fake model for --mock: returns the scripted hypothesis of the clip that is currently being processed."""

    name = "scripted-fake"

    def __init__(self, hypotheses: dict[str, str]) -> None:
        self.hypotheses, self.current = hypotheses, ""

    def transcribe(self, samples):
        from .formats import Segment

        text = self.hypotheses.get(self.current, "")
        return [Segment(0.0, 1.0, text)] if text else []


def load_manifest(clips_dir: Path) -> list[dict]:
    path = Path(clips_dir) / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path} not found. Run: python scripts/download_clips.py") from None
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read {path}: {e}") from None
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of clips")
    for c in data:
        if not isinstance(c, dict) or not all(isinstance(c.get(k), str) and c[k] for k in ("id", "file", "reference")):
            raise ValueError(f"{path}: every clip needs id, file and reference text: {c}")
    return data


def evaluate(clips: list[dict], clips_dir: Path, transcriber, mode: str, timing: bool, hook=None) -> Report:
    report = Report(mode=mode, model=getattr(transcriber, "name", "?"))
    for c in clips:
        if hook:
            hook(c["id"])
        report.add(c["id"], c["reference"], transcribe_file(Path(clips_dir) / c["file"], transcriber), timing)
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="no model, no clips; NOT an accuracy measurement")
    group.add_argument("--live", action="store_true", help="the real model on data/clips (downloads the model on first use)")
    ap.add_argument("--clips", type=Path, default=CLIPS_DIR, help="folder with manifest.json and the audio files")
    ap.add_argument("--model", help="base.en (default) or tiny.en")
    ap.add_argument("--limit", type=int, help="use only the first N clips")
    ap.add_argument("--out", type=Path, help="save the summary plus per-clip details as JSON")
    args = ap.parse_args(argv)

    try:
        if args.mock:
            tmp = Path(tempfile.mkdtemp())
            clips = []
            for i, (cid, ref, _hyp) in enumerate(MOCK_CLIPS):
                syn.write_wav(tmp / f"{cid}.wav", syn.tone(300 + 40 * i, 2.0))
                clips.append({"id": cid, "file": f"{cid}.wav", "reference": ref})
            transcriber = ScriptedTranscriber({cid: hyp for cid, _ref, hyp in MOCK_CLIPS})
            clips_dir, mode, timing = tmp, "mock", False
            print("MODE: mock -- plumbing check with synthetic tones and a scripted fake transcriber. These numbers are NOT accuracy.")
            hook = lambda cid: setattr(transcriber, "current", cid)  # noqa: E731
        else:
            clips, clips_dir, mode, timing, hook = load_manifest(args.clips), args.clips, "live", True, None
            transcriber = FasterWhisperTranscriber(args.model)
            print(f"MODE: live -- the real model '{transcriber.name}' on {len(clips)} clip(s) from {args.clips}.")
        if args.limit:
            clips = clips[: max(args.limit, 1)]
    except (ValueError, ConfigError, OSError) as e:
        print(f"Cannot run evaluation: {e}", file=sys.stderr)
        return 2

    report = evaluate(clips, clips_dir, transcriber, mode, timing, hook)
    summary = report.summary()
    print(json.dumps(summary, indent=2))
    failed = [r for r in report.rows if r.status != "ok"]
    for r in failed:
        print(f"note: clip {r.id} {r.status}: {r.message}", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps({**summary, "clips_detail": [r.as_dict() for r in report.rows]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
