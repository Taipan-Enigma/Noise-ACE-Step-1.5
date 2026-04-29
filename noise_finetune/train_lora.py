#!/usr/bin/env python3
"""
LoRA fine-tune acestep-5Hz-lm-1.7B on the mixed Listener+Planner Merzbow
dataset built by build_training_dataset.py.

    python noise_finetune/train_lora.py --dry-run    # 4 rows, 50 steps sanity
    python noise_finetune/train_lora.py              # the real ~15-30 min run

Saves a LoRA adapter under noise_finetune/lora_output/. Load it at inference
time with:

    base = AutoModelForCausalLM.from_pretrained(...)
    model = PeftModel.from_pretrained(base, "noise_finetune/lora_output/...")
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class PerTaskLossTrainer(Trainer):
    """Trainer that logs loss_listener / loss_planner tags alongside the main loss.

    For batch_size=1 we just tag each step's loss with the sample's _task.
    For batch_size>1 we'd need to split the batch; not needed for this config.
    """

    def __init__(self, *args, **kwargs):
        self._task_buffer: dict[str, list[float]] = {"listener": [], "planner": []}
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tasks = inputs.pop("_task", None)
        ids = inputs.pop("_id", None)  # noqa: F841
        loss_outputs = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss_value = (loss_outputs[0] if return_outputs else loss_outputs).detach().float().item()

        if tasks is not None:
            # `tasks` is a list[str] of length batch_size (all the same if bs=1).
            first = tasks[0] if isinstance(tasks, (list, tuple)) else tasks
            if first in self._task_buffer:
                self._task_buffer[first].append(loss_value)

        return loss_outputs

    def log(self, logs: dict, *args, **kwargs):
        for task, values in self._task_buffer.items():
            if values:
                logs[f"loss_{task}"] = sum(values) / len(values)
                values.clear()
        return super().log(logs, *args, **kwargs)


class StringAwareCollator:
    """Wraps DataCollatorForSeq2Seq but passes _task / _id through untouched."""

    def __init__(self, base_collator):
        self.base = base_collator

    def __call__(self, features):
        tasks = [f.pop("_task", None) for f in features]
        ids = [f.pop("_id", None) for f in features]
        batch = self.base(features)
        batch["_task"] = tasks
        batch["_id"] = ids
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoints" / "acestep-5Hz-lm-1.7B"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "noise_finetune" / "training_data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "noise_finetune" / "lora_output"))
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true",
                        help="Tiny 4-row, 50-step sanity run.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_ds = load_from_disk(str(data_dir / "train"))
    eval_ds = load_from_disk(str(data_dir / "eval"))
    logger.info(f"Train rows: {len(train_ds)} | eval rows: {len(eval_ds)}")

    if args.dry_run:
        # 2 listener + 2 planner rows if we can find them; otherwise first 4.
        listener_idx = [i for i, t in enumerate(train_ds["_task"]) if t == "listener"][:2]
        planner_idx = [i for i, t in enumerate(train_ds["_task"]) if t == "planner"][:2]
        idx = (listener_idx + planner_idx) or list(range(min(4, len(train_ds))))
        train_ds = train_ds.select(idx)
        eval_ds = eval_ds.select(range(min(2, len(eval_ds))))
        logger.info(f"Dry run: reduced to {len(train_ds)} train / {len(eval_ds)} eval")

    logger.info(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {args.model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required with gradient checkpointing / LoRA training
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        training_args = TrainingArguments(
            output_dir=str(output_dir / "dry_run"),
            max_steps=50,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            logging_steps=2,
            save_strategy="no",
            eval_strategy="no",
            bf16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
            remove_unused_columns=False,
        )
    else:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            bf16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
            remove_unused_columns=False,
        )

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    data_collator = StringAwareCollator(base_collator)

    trainer = PerTaskLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    final_dir = output_dir / "checkpoint-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
