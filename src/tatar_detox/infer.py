from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from .baselines.lexical import clean_text


def normalize_generated_text(text: str) -> str:
    text = re.sub(r"([!?.,])\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(model_dir: Path):
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_one(
    text: str,
    tokenizer: T5Tokenizer,
    model: MT5ForConditionalGeneration,
    device: torch.device,
    max_length: int,
    beams: int,
) -> str:
    encoded = tokenizer(
        "detox: " + text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_length=max_length,
            num_beams=beams,
            early_stopping=True,
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mT5 Tatar detoxification inference.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--lexical-postprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, sep="\t")
    required = {"ID", "tat_toxic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required TSV columns: {sorted(missing)}")

    tokenizer, model, device = load_model(args.model)
    predictions: list[str] = []

    for text in frame["tat_toxic"].astype(str):
        output = generate_one(text, tokenizer, model, device, args.max_length, args.beams)
        output = normalize_generated_text(output)
        if args.lexical_postprocess:
            output = clean_text(output)
        predictions.append(output)

    result = pd.DataFrame(
        {"ID": frame["ID"], "tat_toxic": frame["tat_toxic"], "tat_detox1": predictions}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
