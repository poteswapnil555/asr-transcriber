import re

import pytest

from asr_transcriber.formats import Segment, sanitise, timestamp, to_text, to_vtt

S = [Segment(0.0, 3.2, "Hello there."), Segment(3.2, 6.0, "This is a test."), Segment(61.5, 3725.25, "Late cue.")]

# ---- a small WebVTT reader written from the spec, used to check the contract -------------------------
_TIMING = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3}) --> (\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3})$")
_VOICE = re.compile(r"^<v ([^>]+)>(.*)</v>$")


def secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def read_vtt(text):
    """Strict reader: returns [(id, start, end, speaker or None, text)]. Asserts the structure as it goes."""
    assert text.startswith("WEBVTT\n\n"), "header must be WEBVTT followed by a blank line"
    cues = []
    body = text[len("WEBVTT\n\n"):]
    if not body:
        return cues
    assert body.endswith("\n") and not body.endswith("\n\n")
    for block in body.rstrip("\n").split("\n\n"):
        lines = block.split("\n")
        assert len(lines) == 3, f"cue must be id / timing / one line of text: {lines!r}"
        cid, timing, payload = lines
        assert cid.isdigit()
        m = _TIMING.match(timing)
        assert m, f"bad timing line: {timing!r}"
        start, end = secs(*m.groups()[:4]), secs(*m.groups()[4:])
        assert end > start
        assert "-->" not in payload and "<" not in re.sub(r"</?v[^>]*>", "", payload)
        v = _VOICE.match(payload)
        cues.append((int(cid), start, end, v.group(1) if v else None, v.group(2) if v else payload))
    return cues


# ---- plain text --------------------------------------------------------------------------------------
def test_text_has_one_line_per_segment():
    assert to_text(S) == "Hello there.\nThis is a test.\nLate cue.\n"


def test_text_of_nothing_is_empty():
    assert to_text([]) == "" and to_text([Segment(0, 1, "  "), Segment(1, 2, "")]) == ""


# ---- timestamps ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00:00.000"), (1.5, "00:00:01.500"), (59.9996, "00:01:00.000"), (61.001, "00:01:01.001"),
    (3600, "01:00:00.000"), (3725.25, "01:02:05.250"), (360000, "100:00:00.000"), (-5, "00:00:00.000"),
])
def test_timestamp(seconds, expected):
    assert timestamp(seconds) == expected


# ---- VTT contract ----------------------------------------------------------------------------------------
def test_vtt_default_labels_every_cue_as_speaker_unknown():
    cues = read_vtt(to_vtt(S))
    assert [(c[0], c[1], c[2], c[3], c[4]) for c in cues] == [
        (1, 0.0, 3.2, "Speaker ?", "Hello there."), (2, 3.2, 6.0, "Speaker ?", "This is a test."),
        (3, 61.5, 3725.25, "Speaker ?", "Late cue."),
    ]


def test_vtt_without_speaker_has_plain_cue_text():
    text = to_vtt(S, speaker=None)
    assert "<v" not in text and [c[3] for c in read_vtt(text)] == [None, None, None]
    assert "00:00:03.200 --> 00:00:06.000\nThis is a test." in text


def test_vtt_header_and_layout():
    text = to_vtt(S[:1])
    assert text == "WEBVTT\n\n1\n00:00:00.000 --> 00:00:03.200\n<v Speaker ?>Hello there.</v>\n"


def test_vtt_with_no_segments_is_just_a_valid_header():
    assert to_vtt([]) == "WEBVTT\n\n" and read_vtt(to_vtt([])) == []


def test_vtt_escapes_markup_and_removes_arrows():
    text = to_vtt([Segment(0, 1, "a < b & c > d --> e")])
    cue = read_vtt(text)[0]  # the reader asserts there is no "-->" and no raw "<" in the payload
    assert cue[4] == "a &lt; b &amp; c &gt; d -&gt; e"
    assert text.count("-->") == 1  # only the timing line


def test_vtt_cue_text_is_one_line_even_if_the_segment_has_newlines_and_blank_lines():
    text = to_vtt([Segment(0, 1, "first\n\nsecond\r\nthird")])
    assert read_vtt(text)[0][4] == "first second third"


def test_vtt_speaker_label_is_configurable_and_escaped():
    assert read_vtt(to_vtt(S[:1], speaker="Host & Co"))[0][3] == "Host &amp; Co"


def test_vtt_skips_empty_segments_and_numbers_cues_consecutively():
    cues = read_vtt(to_vtt([Segment(0, 1, "one"), Segment(1, 2, "  "), Segment(2, 3, "two")]))
    assert [(c[0], c[4]) for c in cues] == [(1, "one"), (2, "two")]


@pytest.mark.parametrize("start,end", [(5.0, 5.0), (5.0, 4.0), (5.0, float("nan")), (float("inf"), 9.0), (-3.0, -1.0)])
def test_vtt_bad_times_are_repaired_so_every_cue_ends_after_it_starts(start, end):
    cues = read_vtt(to_vtt([Segment(start, end, "text here")]))
    assert cues[0][2] > cues[0][1] >= 0


def test_vtt_start_times_never_go_backwards():
    cues = read_vtt(to_vtt([Segment(10, 12, "a"), Segment(5, 6, "b"), Segment(11, 13, "c")]))
    starts = [c[1] for c in cues]
    assert starts == sorted(starts)


def test_vtt_millisecond_rounding_never_prints_1000():
    text = to_vtt([Segment(0.9996, 1.9999, "x")])
    assert ".1000" not in text and "00:00:01.000 --> 00:00:02.000" in text


def test_vtt_handles_a_very_long_recording():
    text = to_vtt([Segment(i * 3.0, i * 3.0 + 2.5, f"cue number {i}") for i in range(5000)])
    cues = read_vtt(text)
    assert len(cues) == 5000 and cues[-1][0] == 5000


def test_sanitise_keeps_order_and_drops_empties():
    out = sanitise([Segment(1, 2, "a"), Segment(0, 1, ""), Segment(3, 2, "b")])
    assert [(s.text, s.start, s.end) for s in out] == [("a", 1, 2), ("b", 3, 3)]
