"""Download a few LibriSpeech test-clean clips with their reference transcripts into data/clips/.

LibriSpeech is a third-party corpus: the clips are NOT part of this repository (data/clips/ is git-ignored).
Source:  https://www.openslr.org/12/   (test-clean.tar.gz, 346 MB)
Licence: CC BY 4.0. Credit: "LibriSpeech: an ASR corpus based on public domain audio books",
         Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur, ICASSP 2015.

The archive is STREAMED and reading stops as soon as N clips with transcripts have been found, so only the start of the
archive (a few MB, not 346 MB) should be downloaded. Nothing is written outside data/clips/.

Usage:  python scripts/download_clips.py [--n 10] [--force]
Output: data/clips/<id>.flac and data/clips/manifest.json  [{"id", "file", "reference"}]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zlib
from pathlib import Path

MIRRORS = [
    "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
    "https://openslr.elda.org/resources/12/test-clean.tar.gz",
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
]
DEST = Path("data/clips")
MAX_READ_BYTES = 200 * 1024 * 1024  # never read more than this from the stream
MAX_CLIP_BYTES = 5 * 1024 * 1024
_MEMBER = re.compile(r"^(?:\./)?LibriSpeech/test-clean/(\d+)/(\d+)/(\1-\2-\d{4}\.flac|\1-\2\.trans\.txt)$")
_ID = re.compile(r"^\d+-\d+-\d{4}$")


class _Capped:
    """File-like wrapper that stops a runaway download."""

    def __init__(self, raw, limit: int) -> None:
        self.raw, self.limit, self.read_bytes = raw, limit, 0

    def read(self, n: int = -1) -> bytes:
        data = self.raw.read(n)
        self.read_bytes += len(data)
        if self.read_bytes > self.limit:
            raise OSError(f"stopped: read more than {self.limit // 1024 // 1024} MB without finding enough clips")
        return data


def collect(stream, n: int) -> list[tuple[str, bytes, str]]:
    """Read a test-clean tar.gz stream; return up to n (id, flac bytes, reference text), in archive order."""
    flacs: dict[str, bytes] = {}
    refs: dict[str, str] = {}
    order: list[str] = []
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            m = _MEMBER.match(member.name) if member.isfile() else None
            if not m:
                continue
            name = m.group(3)
            if name.endswith(".flac"):
                if member.size > MAX_CLIP_BYTES:
                    continue
                flacs[name[:-5]] = tar.extractfile(member).read()
                order.append(name[:-5])
            else:
                for line in tar.extractfile(member).read().decode("utf-8", errors="replace").splitlines():
                    cid, _, text = line.strip().partition(" ")
                    if _ID.match(cid) and text.strip():
                        refs[cid] = text.strip()
            ready = [i for i in order if i in refs]
            if len(ready) >= n:
                break
    return [(i, flacs[i], refs[i]) for i in order if i in refs][:n]


def download(dest: Path, n: int = 10, force: bool = False, mirrors: list[str] | None = None) -> int:
    manifest = dest / "manifest.json"
    if manifest.exists() and not force:
        print(f"Already present: {manifest} (use --force to download again)")
        return 0
    last_error = "no mirror could be reached"
    for url in mirrors or MIRRORS:
        req = urllib.request.Request(url, headers={"User-Agent": "asr-transcriber-download/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                clips = collect(_Capped(resp, MAX_READ_BYTES), n)
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = f"{url}: {e}"
            continue
        except (tarfile.TarError, EOFError, zlib.error, OSError) as e:
            last_error = f"{url}: the download failed or the archive is corrupt ({type(e).__name__}: {e})"
            continue
        if not clips:
            last_error = f"{url}: no clips with transcripts were found in the archive"
            continue
        dest.mkdir(parents=True, exist_ok=True)
        entries = []
        for cid, data, text in clips:
            (dest / f"{cid}.flac").write_bytes(data)
            entries.append({"id": cid, "file": f"{cid}.flac", "reference": text})
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Saved {len(entries)} clips and {manifest}. Credit: LibriSpeech (Panayotov et al., ICASSP 2015), CC BY 4.0, https://www.openslr.org/12/")
        if len(entries) < n:
            print(f"Note: only {len(entries)} clips found before the archive ended (asked for {n}).", file=sys.stderr)
        return 0
    print(f"Download failed: {last_error}", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="number of clips (default 10)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.n < 1 or args.n > 100:
        print("--n must be between 1 and 100", file=sys.stderr)
        return 2
    return download(DEST, args.n, args.force)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
