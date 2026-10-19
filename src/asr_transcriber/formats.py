"""Plain-text and WebVTT output.

WebVTT written here is plain and strict, so other tools can read it:
  WEBVTT, a blank line, then numbered cues:   1 / 00:00:01.000 --> 00:00:03.500 / <v Speaker ?>text</v>
- timestamps always have hours (HH:MM:SS.mmm), are never negative, and each cue ends after it starts
- cue text is one line, with & < > escaped and "-->" removed, so it can never break the cue structure
- Whisper does not identify speakers. By default every cue is labelled "Speaker ?" (a voice tag), which is honest but makes
  tools that merge consecutive cues of one speaker (such as meeting-summarizer) see one long turn.
  With speaker=None (CLI: --no-speaker) cues carry no speaker, so each stays a separate turn with its own timestamp.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

DEFAULT_SPEAKER = "Speaker ?"


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float
    text: str


def clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\x00", "")).strip()


def sanitise(segments: list[Segment]) -> list[Segment]:
    """Drop empty segments, fix bad times (negative, NaN, end before start) and keep the order non-decreasing."""
    out: list[Segment] = []
    last_start = 0.0
    for seg in segments:
        text = clean(seg.text)
        if not text:
            continue
        start = seg.start if isinstance(seg.start, (int, float)) and math.isfinite(seg.start) else last_start
        end = seg.end if isinstance(seg.end, (int, float)) and math.isfinite(seg.end) else start
        start = max(start, last_start, 0.0)
        end = max(end, start)
        out.append(Segment(start, end, text))
        last_start = start
    return out


def to_text(segments: list[Segment]) -> str:
    """One line per segment."""
    lines = [s.text for s in sanitise(segments)]
    return "\n".join(lines) + ("\n" if lines else "")


def timestamp(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def _escape(text: str) -> str:
    text = text.replace("-->", "->")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(segments: list[Segment], speaker: str | None = DEFAULT_SPEAKER) -> str:
    cues = []
    for n, seg in enumerate(sanitise(segments), start=1):
        start, end = timestamp(seg.start), timestamp(max(seg.end, seg.start + 0.001))
        body = _escape(seg.text)
        if speaker:
            who = _escape(clean(speaker)).replace("\n", " ")
            body = f"<v {who}>{body}</v>"
        cues.append(f"{n}\n{start} --> {end}\n{body}")
    return "WEBVTT\n\n" + "\n\n".join(cues) + ("\n" if cues else "")
