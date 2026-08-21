from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import sacrebleu


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    pred_counts = Counter(tokenize(prediction))
    ref_counts = Counter(tokenize(reference))
    overlap = sum((pred_counts & ref_counts).values())

    if not pred_counts and not ref_counts:
        return 1.0
    if not pred_counts or not ref_counts:
        return 0.0

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def character_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, prediction, reference).ratio()


def infer_column(rows: list[dict[str, str]], candidates: tuple[str, ...]) -> str:
    if not rows:
        raise ValueError("TSV file is empty.")
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise ValueError(f"Expected one of columns: {', '.join(candidates)}")


def evaluate(
    reference_path: Path,
    prediction_path: Path,
    id_column: str = "ID",
) -> dict[str, float]:
    references = read_tsv(reference_path)
    predictions = read_tsv(prediction_path)

    ref_input_col = infer_column(references, ("tat_toxic", "input"))
    ref_output_col = infer_column(references, ("tat_detox1", "target", "output", "prediction"))
    pred_input_col = infer_column(predictions, ("tat_toxic", "input"))
    pred_output_col = infer_column(predictions, ("tat_detox1", "output", "prediction"))

    ref_by_id = {str(row[id_column]): row for row in references}
    pred_by_id = {str(row[id_column]): row for row in predictions}

    if set(ref_by_id) != set(pred_by_id):
        raise ValueError("Reference and prediction files contain different ID sets.")

    ids = sorted(ref_by_id, key=lambda value: int(value) if value.isdigit() else value)
    sources = [ref_by_id[item][ref_input_col] for item in ids]
    gold = [ref_by_id[item][ref_output_col] for item in ids]
    outputs = [pred_by_id[item][pred_output_col] for item in ids]
    pred_sources = [pred_by_id[item][pred_input_col] for item in ids]

    if pred_sources != sources:
        raise ValueError("Prediction inputs are not aligned exactly with the reference inputs.")

    size = len(ids)
    return {
        "rows": float(size),
        "exact_match": sum(pred == ref for pred, ref in zip(outputs, gold)) / size,
        "chrf_pp": sacrebleu.corpus_chrf(outputs, [gold], word_order=2).score,
        "char_similarity_to_gold": sum(
            character_similarity(pred, ref) for pred, ref in zip(outputs, gold)
        ) / size,
        "token_f1_to_gold": sum(token_f1(pred, ref) for pred, ref in zip(outputs, gold)) / size,
        "source_similarity": sum(
            character_similarity(pred, source) for pred, source in zip(outputs, sources)
        ) / size,
        "changed_fraction": sum(pred != source for pred, source in zip(outputs, sources)) / size,
        "mean_length_ratio": sum(
            len(pred) / len(source) if source else 1.0
            for pred, source in zip(outputs, sources)
        ) / size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Tatar detoxification TSV against paired human references."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--id-column", default="ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.reference, args.prediction, args.id_column)

    print(f"Rows: {int(metrics['rows'])}")
    print(f"Exact match: {metrics['exact_match']:.3%}")
    print(f"chrF++: {metrics['chrf_pp']:.2f}")
    print(f"Character similarity to gold: {metrics['char_similarity_to_gold']:.4f}")
    print(f"Token F1 to gold: {metrics['token_f1_to_gold']:.4f}")
    print(f"Source similarity: {metrics['source_similarity']:.4f}")
    print(f"Rows changed: {metrics['changed_fraction']:.1%}")
    print(f"Mean output/input length ratio: {metrics['mean_length_ratio']:.3f}")


if __name__ == "__main__":
    main()
