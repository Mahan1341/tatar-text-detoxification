from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from .lexicons import AGGRESSIVE_PHRASES, NOISE_PREFIXES, TOXIC_TERMS

TATAR_CHARS = set("әөүҗңһӘӨҮҖҢҺ")


def split_into_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]


def is_clean_sentence(text: str) -> bool:
    text = text.strip()
    if not 40 <= len(text) <= 150:
        return False

    words = text.split()
    if not 6 <= len(words) <= 25:
        return False

    if re.search(r"\b(РФ|USD|EUR|cm|km|COVID)\b", text):
        return False
    if re.search(r"[\*\[\]\(\)\{\}\|<>]", text):
        return False
    if sum(char in TATAR_CHARS for char in text) < 2:
        return False
    if re.search(r"(глава|совет|комитет|правительство)", text.lower()):
        return False
    if text.isupper() or "--" in text:
        return False
    if len(set(words)) < 0.6 * len(words):
        return False

    return True


def inject_toxicity(sentence: str, rng: random.Random) -> str:
    words = sentence.split()
    index = rng.randrange(len(words))
    toxic = rng.choice(TOXIC_TERMS)

    if rng.random() < 0.5:
        words[index] = toxic
    else:
        words.insert(index, toxic)

    if rng.random() < 0.15:
        words.insert(0, rng.choice(AGGRESSIVE_PHRASES))
    if rng.random() < 0.25:
        words.append(rng.choice(AGGRESSIVE_PHRASES))
    if rng.random() < 0.30:
        cap_index = rng.randrange(len(words))
        words[cap_index] = words[cap_index].upper()
    if rng.random() < 0.25:
        words.append(rng.choice(["!!!", "??!", "!?!!", "...", "--"]))
    if rng.random() < 0.12 and len(words) > 6:
        swap_index = rng.randint(1, len(words) - 3)
        words[swap_index], words[swap_index + 1] = words[swap_index + 1], words[swap_index]
    if rng.random() < 0.15:
        words.insert(rng.randint(0, len(words)), rng.choice(TOXIC_TERMS))

    text = " ".join(words)
    if rng.random() < 0.10:
        text = re.sub(r"\s+", "  ", text)
    if rng.random() < 0.05:
        text = text.replace(" ", "   ")
    if rng.random() < 0.07:
        text = f"{rng.choice(NOISE_PREFIXES)} {text}"
    if rng.random() < 0.05 and len(words) > 4:
        parts = text.split()
        parts.insert(rng.randint(2, len(parts) - 2), rng.choice(AGGRESSIVE_PHRASES))
        text = " ".join(parts)

    return text


def build_pairs(sample_size: int, seed: int) -> list[dict[str, str]]:
    dataset = load_dataset("veryrealtatarperson/tt-azatliq-crawl", split="news_clean")

    clean_sentences: list[str] = []
    for row in dataset:
        for sentence in split_into_sentences(row["article_text"]):
            if is_clean_sentence(sentence):
                clean_sentences.append(sentence)

    if sample_size > len(clean_sentences):
        raise ValueError(
            f"Requested {sample_size} samples, but only {len(clean_sentences)} passed filtering."
        )

    rng = random.Random(seed)
    sampled = rng.sample(clean_sentences, sample_size)
    return [
        {"input": inject_toxicity(sentence, rng), "target": sentence}
        for sentence in sampled
    ]


def write_jsonl(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic Tatar detoxification pairs.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic.jsonl"))
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_pairs(args.sample_size, args.seed)
    write_jsonl(rows, args.output)
    print(f"Saved {len(rows)} pairs to {args.output}")


if __name__ == "__main__":
    main()
