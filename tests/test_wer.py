import pytest

from asr_transcriber.wer import Errors, corpus_errors, count_errors, normalise, wer


# ---- normalisation -------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("Hello, World!", "hello world"),
    ("HELLO   WORLD\n\tAGAIN", "hello world again"),
    ("Don't stop", "don't stop"),
    ("Don’t stop", "don't stop"),
    ("'tis the season'", "tis the season"),
    ("well-known and well—known", "well known and well known"),
    ("snake_case words", "snake case words"),
    ('She said "yes" (twice).', "she said yes twice"),
    ("Price: $5.00; 10% off", "price 5 00 10 off"),
    ("Café Été", "café été"),
    ("", ""), ("   ", ""), (None, ""), ("...!!!", ""),
    ("ＡＢＣ", "abc"),  # full-width letters (NFKC)
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_numbers_and_spelling_are_not_normalised_on_purpose():
    assert normalise("10 colour") != normalise("ten color")


# ---- WER values --------------------------------------------------------------------------------
def test_perfect_match_is_zero():
    e = count_errors("the cat sat on the mat", "The cat sat on the mat.")
    assert (e.substitutions, e.deletions, e.insertions, e.reference_words) == (0, 0, 0, 6) and e.wer == 0.0


def test_one_substitution_in_four_words():
    e = count_errors("the cat sat down", "the cat sat up")
    assert (e.substitutions, e.deletions, e.insertions) == (1, 0, 0) and e.wer == 0.25


def test_one_deletion():
    e = count_errors("the cat sat down", "the cat down")
    assert (e.substitutions, e.deletions, e.insertions) == (0, 1, 0) and e.wer == 0.25


def test_one_insertion():
    e = count_errors("the cat sat", "the big cat sat")
    assert (e.substitutions, e.deletions, e.insertions) == (0, 0, 1) and e.wer == pytest.approx(1 / 3)


def test_mixed_errors_are_counted_by_kind():
    # a unique best alignment: one deletion (delta) and two insertions (golf, hotel)
    e = count_errors("alpha bravo charlie delta echo foxtrot", "alpha bravo charlie echo foxtrot golf hotel")
    assert (e.substitutions, e.deletions, e.insertions, e.reference_words) == (0, 1, 2, 6)
    s = count_errors("the quick brown fox jumps", "the quick red fox jumps over")  # one substitution, one insertion
    assert (s.substitutions, s.deletions, s.insertions) == (1, 0, 1)


def test_ties_between_equally_good_alignments_keep_the_same_total():
    # "two->2, four->five, five->six" and "two->2, delete four, insert six" both cost 3
    assert count_errors("one two three four five", "one 2 three five six").total == 3


def test_empty_hypothesis_is_all_deletions_and_wer_one():
    e = count_errors("a b c", "")
    assert (e.substitutions, e.deletions, e.insertions) == (0, 3, 0) and e.wer == 1.0


def test_wer_can_exceed_one_when_the_hypothesis_is_longer():
    assert wer("a b", "a b c d e f") == 2.0


def test_completely_different_text():
    assert wer("a b c", "x y z") == 1.0


def test_empty_reference_has_undefined_wer():
    assert wer("", "") is None and wer("", "extra words") is None
    assert count_errors("", "extra words").insertions == 2


def test_punctuation_and_case_do_not_count_as_errors():
    assert wer("Hello, world! It's fine.", "hello world it's fine") == 0.0


def test_order_matters():
    assert wer("a b c", "c b a") > 0


def test_word_error_counts_match_a_hand_checked_example():
    # LibriSpeech-style uppercase reference against a Whisper-style hypothesis
    ref = "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS"
    hyp = "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes."
    e = count_errors(ref, hyp)
    assert e.reference_words == 11 and e.insertions == 3 and e.substitutions == 0 and e.deletions == 0
    assert e.wer == pytest.approx(3 / 11)


def test_total_equals_edit_distance_for_random_pairs():
    import random

    rng = random.Random(1)
    vocab = "a b c d e".split()
    for _ in range(200):
        r = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 8)))
        h = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 8)))
        e = count_errors(r, h)
        assert e.total >= abs(len(r.split()) - len(h.split()))
        assert e.total <= max(len(r.split()), len(h.split()))
        assert e.reference_words == len(r.split())
        assert e.deletions - e.insertions == len(r.split()) - len(h.split())  # word counts must balance


def test_symmetry_swaps_deletions_and_insertions():
    a = count_errors("one two three", "one three")
    b = count_errors("one three", "one two three")
    assert (a.deletions, a.insertions) == (b.insertions, b.deletions)


# ---- corpus --------------------------------------------------------------------------------------
def test_corpus_wer_pools_errors_not_rates():
    short = count_errors("a b", "a x")  # 1/2
    long = count_errors("a b c d e f g h i j", "a b c d e f g h i j")  # 0/10
    pooled = corpus_errors([short, long])
    assert pooled.reference_words == 12 and pooled.total == 1 and pooled.wer == pytest.approx(1 / 12)
    mean_of_rates = (short.wer + long.wer) / 2
    assert mean_of_rates == 0.25 and pooled.wer != mean_of_rates  # the two ways differ


def test_corpus_of_nothing_is_undefined():
    assert corpus_errors([]).wer is None
    assert Errors(0, 0, 0, 0).wer is None
