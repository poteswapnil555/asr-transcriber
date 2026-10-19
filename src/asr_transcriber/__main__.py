"""Usage: python -m asr_transcriber AUDIO_FILE [--model base.en] [--out-dir output] [--no-speaker] [--print]

Writes <out-dir>/<name>.txt and <out-dir>/<name>.vtt (existing files are overwritten).
Exit code: 0 transcribed, 1 no speech or an error (message on stderr), 2 command-line usage error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError
from .formats import DEFAULT_SPEAKER, to_text, to_vtt
from .pipeline import transcribe_file
from .transcriber import FasterWhisperTranscriber, Transcriber


def make_transcriber(model: str | None) -> Transcriber:
    return FasterWhisperTranscriber(model)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="wav, mp3, flac, ogg or m4a")
    ap.add_argument("--model", help="base.en (default) or tiny.en; also ASR_MODEL")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--no-speaker", action="store_true", help='leave the "Speaker ?" label out of the VTT cues')
    ap.add_argument("--speaker-label", default=DEFAULT_SPEAKER, help='label used in the VTT (default "Speaker ?"); Whisper does not know speaker names')
    ap.add_argument("--print", action="store_true", help="also print the text")
    args = ap.parse_args(argv)

    try:
        transcriber = make_transcriber(args.model)
        result = transcribe_file(args.audio, transcriber)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    if result.status != "ok":
        print(f"{result.status}: {result.message}", file=sys.stderr)
        return 1

    speaker = None if args.no_speaker else args.speaker_label
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        txt, vtt = args.out_dir / f"{args.audio.stem}.txt", args.out_dir / f"{args.audio.stem}.vtt"
        txt.write_text(to_text(result.segments), encoding="utf-8")
        vtt.write_text(to_vtt(result.segments, speaker), encoding="utf-8")
    except OSError as e:
        print(f"error: could not write the output files ({e})", file=sys.stderr)
        return 1
    rtf = f", {result.real_time_factor:.2f}x real time" if result.real_time_factor is not None else ""
    print(f"ok: {len(result.segments)} segments from {result.duration:.1f} s of audio{rtf} (model {transcriber.name})")
    print(f"text: {txt}\nvtt:  {vtt}")
    if args.print:
        print(to_text(result.segments), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
