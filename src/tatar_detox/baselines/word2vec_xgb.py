from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .lexical import clean_text


def embed_sentence(model: Word2Vec, text: str) -> np.ndarray:
    vectors = [model.wv[word] for word in text.split() if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def load_pairs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if "input" not in row or "target" not in row:
                raise ValueError("Each JSONL row must contain 'input' and 'target'.")
            rows.append(row)
    return rows


def expand_pairs(pairs: list[dict[str, str]], model: Word2Vec) -> tuple[np.ndarray, np.ndarray]:
    texts = [row["input"] for row in pairs] + [row["target"] for row in pairs]
    labels = np.array([1] * len(pairs) + [0] * len(pairs), dtype=np.int64)
    features = np.vstack([embed_sentence(model, text) for text in texts])
    return features, labels


def train_gate(pairs_path: Path, model: Word2Vec, seed: int) -> XGBClassifier:
    pairs = load_pairs(pairs_path)
    train_pairs, _ = train_test_split(pairs, test_size=0.1, random_state=seed)
    x_train, y_train = expand_pairs(train_pairs, model)

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
    )
    classifier.fit(x_train, y_train)
    return classifier


def detox_text(text: str, model: Word2Vec, classifier: XGBClassifier) -> str:
    vector = embed_sentence(model, text).reshape(1, -1)
    return clean_text(text) if int(classifier.predict(vector)[0]) == 1 else text


def run(args: argparse.Namespace) -> None:
    model = Word2Vec.load(str(args.word2vec))
    classifier = train_gate(args.pairs, model, args.seed)

    frame = pd.read_csv(args.input, sep="\t")
    required = {"ID", "tat_toxic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required TSV columns: {sorted(missing)}")

    predictions = [detox_text(str(text), model, classifier) for text in frame["tat_toxic"]]
    result = pd.DataFrame(
        {"ID": frame["ID"], "tat_toxic": frame["tat_toxic"], "tat_detox1": predictions}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tatar2Vec + XGBoost baseline.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--word2vec", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
