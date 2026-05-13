"""
Task-01: Text Generation with GPT-2
=====================================
Fine-tune GPT-2 on a custom dataset to generate coherent and contextually
relevant text that mimics the style and structure of your training data.

References:
  #1 - HuggingFace Transformers: https://huggingface.co/docs/transformers
  #2 - GPT-2 Paper: https://openai.com/research/language-unsupervised
"""

import os
import math
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class TextDataset(Dataset):
    """
    Tokenises a plain-text file and breaks it into overlapping
    blocks of `block_size` tokens, ready for causal LM training.
    """

    def __init__(self, file_path: str, tokenizer: GPT2Tokenizer, block_size: int = 128):
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        logger.info(f"Loading dataset from: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized = tokenizer.encode(text, add_special_tokens=True)
        logger.info(f"Total tokens in dataset: {len(tokenized):,}")

        # Build non-overlapping blocks
        self.examples = []
        for i in range(0, len(tokenized) - block_size + 1, block_size):
            self.examples.append(torch.tensor(tokenized[i : i + block_size], dtype=torch.long))

        logger.info(f"Total training examples: {len(self.examples):,}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Tokenizer & Model ──────────────────────
    logger.info(f"Loading GPT-2 tokenizer & model ({args.model_name}) …")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token by default

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)

    # ── Dataset & DataLoader ───────────────────
    dataset = TextDataset(args.train_file, tokenizer, block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ── Optimizer & Scheduler ──────────────────
    total_steps = len(dataloader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── Training Loop ──────────────────────────
    logger.info("Starting fine-tuning …\n")
    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader, 1):
            batch = batch.to(device)

            # For causal LM: labels == input_ids (model shifts internally)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                avg = epoch_loss / step
                ppl = math.exp(avg) if avg < 20 else float("inf")
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss: {avg:.4f} | Perplexity: {ppl:.2f}"
                )

        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(
            f"\n✓ Epoch {epoch} complete — "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Perplexity: {math.exp(min(avg_epoch_loss, 20)):.2f}\n"
        )

    # ── Save ───────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to: {output_dir}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 on a custom text dataset")
    p.add_argument("--train_file",  default="data/train.txt",  help="Path to training .txt file")
    p.add_argument("--model_name",  default="gpt2",            help="HuggingFace model id (gpt2 / gpt2-medium / …)")
    p.add_argument("--output_dir",  default="output/model",    help="Where to save the fine-tuned model")
    p.add_argument("--epochs",      type=int,   default=3,     help="Number of training epochs")
    p.add_argument("--batch_size",  type=int,   default=4,     help="Training batch size")
    p.add_argument("--block_size",  type=int,   default=128,   help="Token block size for each training example")
    p.add_argument("--lr",          type=float, default=5e-5,  help="Learning rate")
    p.add_argument("--log_every",   type=int,   default=10,    help="Log every N steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
