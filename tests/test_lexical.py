from tatar_detox.baselines.lexical import clean_text


def test_clean_text_removes_known_toxic_term() -> None:
    text = "Бу кеше сука түгел"
    cleaned = clean_text(text)
    assert "сука" not in cleaned.lower()


def test_clean_text_preserves_neutral_text() -> None:
    text = "Мин бүген университетка барам"
    assert clean_text(text) == text
