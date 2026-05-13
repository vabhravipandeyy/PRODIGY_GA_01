"""
Task-01: Text Generation with GPT-2 — Inference
================================================
Generate text from your fine-tuned model using various decoding strategies.
"""

import argparse
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from: {args.model_dir}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.eval()
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    logger.info(f"\nPrompt : {args.prompt!r}")
    logger.info(f"Strategy: {args.strategy}\n")
    logger.info("=" * 60)

    with torch.no_grad():
        if args.strategy == "greedy":
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        elif args.strategy == "beam":
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elif args.strategy == "sampling":
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n" + generated + "\n")
    print("=" * 60)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(generated)
        logger.info(f"Output saved to: {args.output_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with a fine-tuned GPT-2 model")
    p.add_argument("--model_dir",      default="output/model",         help="Fine-tuned model directory")
    p.add_argument("--prompt",         default="Once upon a time",      help="Text prompt")
    p.add_argument("--strategy",       default="sampling",
                   choices=["greedy", "beam", "sampling"],              help="Decoding strategy")
    p.add_argument("--max_new_tokens", type=int,   default=200,         help="Max tokens to generate")
    p.add_argument("--temperature",    type=float, default=0.9,         help="Sampling temperature (0.1–2.0)")
    p.add_argument("--top_k",          type=int,   default=50,          help="Top-K sampling")
    p.add_argument("--top_p",          type=float, default=0.95,        help="Top-P (nucleus) sampling")
    p.add_argument("--num_beams",      type=int,   default=5,           help="Beam width (beam search only)")
    p.add_argument("--output_file",    default=None,                    help="Optional: save generated text to file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
