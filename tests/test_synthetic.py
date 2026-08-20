import random

from tatar_detox.synthetic import inject_toxicity, is_clean_sentence


def test_inject_toxicity_changes_sentence() -> None:
    sentence = "Бу татар телендә язылган шактый озын һәм чиста җөмлә булып тора."
    corrupted = inject_toxicity(sentence, random.Random(42))
    assert corrupted != sentence


def test_filter_accepts_reasonable_tatar_sentence() -> None:
    sentence = "Бу татар телендә язылган шактый озын һәм мәгънәле җөмлә булып тора."
    assert is_clean_sentence(sentence)
