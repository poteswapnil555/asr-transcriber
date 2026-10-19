"""Word error rate, written from scratch (no jiwer) and tested.

WER = (substitutions + deletions + insertions) / words in the reference, from a word-level edit distance.

Normalisation (applied to BOTH reference and hypothesis, so formatting differences are not counted as errors):
1. Unicode NFKC, then lowercase
2. curly apostrophes become straight ones
3. hyphens, dashes and underscores become spaces ("well-known" -> "well known")
4. every other character that is not a letter, digit or apostrophe is removed (punctuation, quotes, symbols)
5. apostrophes at the start or end of a word are removed ("'tis" -> "tis"); inside a word they stay ("don't")
6. whitespace is collapsed
NOT normalised (these count as errors): numbers vs spelled-out numbers ("10" vs "ten"), British vs American spelling,
contractions vs full forms ("do not" vs "don't"), filler words. This keeps the metric simple but makes it stricter than the
usual Whisper-style normaliser, so scores are not directly comparable with published numbers.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalise(text: object) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = re.sub(r"[-‐-―_]", " ", s)
    s = re.sub(r"[^\w' ]", " ", s, flags=re.UNICODE)
    words = [w.strip("'") for w in s.split()]
    return " ".join(w for w in words if w)


@dataclass(frozen=True)
class Errors:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float | None:
        """None when the reference has no words (the rate is undefined)."""
        return self.total / self.reference_words if self.reference_words else None


def count_errors(reference: object, hypothesis: object) -> Errors:
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    n, m = len(ref), len(hyp)
    # dist[i][j] = edit distance between ref[:i] and hyp[:j]
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j - 1] + cost, dist[i - 1][j] + 1, dist[i][j - 1] + 1)
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:  # walk back to count each kind of error
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return Errors(s, d, ins, n)


def wer(reference: object, hypothesis: object) -> float | None:
    return count_errors(reference, hypothesis).wer


def corpus_errors(pairs: list[Errors]) -> Errors:
    """Pool the errors of many clips. Corpus WER = total errors / total reference words (not the mean of per-clip rates)."""
    return Errors(sum(e.substitutions for e in pairs), sum(e.deletions for e in pairs),
                  sum(e.insertions for e in pairs), sum(e.reference_words for e in pairs))
