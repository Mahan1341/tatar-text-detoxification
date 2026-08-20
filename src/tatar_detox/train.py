from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "google/mt5-small"


def load_jsonl(path: Path) -> Dataset:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if "input" not in row or "target" not in row:
                raise ValueError("Each JSONL row must contain 'input' and 'target'.")
            rows.append(row)
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer: T5Tokenizer, max_length: int) -> Dataset:
    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        model_inputs = tokenizer(
            ["detox: " + text for text in batch["input"]],
            max_length=max_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mT5-small for Tatar detoxification.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.config.use_cache = False

    if args.eval_data:
        train_raw = load_jsonl(args.train_data)
        eval_raw = load_jsonl(args.eval_data)
    else:
        split = load_jsonl(args.train_data).train_test_split(test_size=0.05, seed=args.seed)
        train_raw = split["train"]
        eval_raw = split["test"]

    train_dataset = tokenize_dataset(train_raw, tokenizer, args.max_length)
    eval_dataset = tokenize_dataset(eval_raw, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
