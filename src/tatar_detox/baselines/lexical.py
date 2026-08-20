from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import pandas as pd

from ..lexicons import SOFT_REPLACEMENTS, TOXIC_TERMS


def clean_text(text: str) -> str:
    words = text.split()
    output: list[str] = []

    for word in words:
        normalized = word.lower().strip(string.punctuation)
        matched = next((term for term in TOXIC_TERMS if term in normalized), None)

        if matched is None:
            output.append(word)
            continue

        replacement = next(
            (value for key, value in SOFT_REPLACEMENTS.items() if key in normalized),
            None,
        )
        if replacement:
            output.append(replacement)

    return re.sub(r"\s+", " ", " ".join(output)).strip()


def run_tsv(input_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path, sep="\t")
    required = {"ID", "tat_toxic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required TSV columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "ID": frame["ID"],
            "tat_toxic": frame["tat_toxic"],
            "tat_detox1": frame["tat_toxic"].astype(str).map(clean_text),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lexical detoxification baseline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tsv(args.input, args.output)


if __name__ == "__main__":
    main()
