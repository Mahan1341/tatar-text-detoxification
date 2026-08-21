from pathlib import Path

from tatar_detox.evaluate import evaluate, token_f1


def test_token_f1_identical_text() -> None:
    assert token_f1("сәлам дөнья", "сәлам дөнья") == 1.0


def test_evaluate_aligned_tsv(tmp_path: Path) -> None:
    reference = tmp_path / "reference.tsv"
    prediction = tmp_path / "prediction.tsv"

    reference.write_text(
        "ID\ttat_toxic\ttat_detox1\n"
        "0\tсука кеше\tначар кеше\n"
        "1\tмин өйгә барам\tмин өйгә барам\n",
        encoding="utf-8",
    )
    prediction.write_text(
        "ID\ttat_toxic\ttat_detox1\n"
        "0\tсука кеше\tначар кеше\n"
        "1\tмин өйгә барам\tмин өйгә барам\n",
        encoding="utf-8",
    )

    metrics = evaluate(reference, prediction)

    assert metrics["rows"] == 2.0
    assert metrics["exact_match"] == 1.0
    assert metrics["chrf_pp"] == 100.0
    assert metrics["token_f1_to_gold"] == 1.0
