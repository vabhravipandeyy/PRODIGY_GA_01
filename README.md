# Task-01 · Text Generation with GPT-2

> Fine-tune GPT-2 on a custom dataset to generate coherent and contextually
> relevant text that mimics the style and structure of your training data.

---

## Project Structure

```
gpt2_text_gen/
├── prepare_data.py   # Clean & split your corpus into train/val
├── train.py          # Fine-tune GPT-2 on data/train.txt
├── generate.py       # Generate text from the fine-tuned model
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Prepare data

**Option A — use the built-in sample corpus (demo)**
```bash
python prepare_data.py
# Creates data/train.txt and data/val.txt automatically
```

**Option B — use your own text file**
```bash
python prepare_data.py --input_file my_corpus.txt --output_dir data
```

### 3 · Fine-tune GPT-2
```bash
python train.py \
  --train_file data/train.txt \
  --model_name gpt2 \
  --output_dir output/model \
  --epochs 3 \
  --batch_size 4 \
  --block_size 128 \
  --lr 5e-5
```

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `gpt2` | Base model (`gpt2`, `gpt2-medium`, `gpt2-large`) |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `4` | Batch size (lower if OOM) |
| `--block_size` | `128` | Token context window per example |
| `--lr` | `5e-5` | Learning rate |

### 4 · Generate text
```bash
# Nucleus sampling (recommended)
python generate.py \
  --model_dir output/model \
  --prompt "The future of AI is" \
  --strategy sampling \
  --temperature 0.9 \
  --top_p 0.95 \
  --max_new_tokens 200

# Beam search (more deterministic)
python generate.py \
  --model_dir output/model \
  --prompt "Once upon a time" \
  --strategy beam \
  --num_beams 5
```

---

## Decoding Strategies

| Strategy | Flag | When to use |
|---|---|---|
| Greedy | `--strategy greedy` | Fast baseline; tends to repeat |
| Beam Search | `--strategy beam` | More coherent; less creative |
| Nucleus (Top-P) | `--strategy sampling` | Best for creative, varied text |

### Temperature guide
| Value | Effect |
|---|---|
| `0.3` | Conservative, repetitive |
| `0.7` | Balanced |
| `0.9` | Creative (recommended) |
| `1.2+` | Very random / experimental |

---

## Tips

- Start with **gpt2** (117M params) — fast and works on CPU.
- Use **gpt2-medium** (345M) for higher-quality outputs if you have a GPU.
- Training loss should drop below **2.5** for good generation quality.
- Lower perplexity = model fits the training data better.

---

## References

1. [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
2. [Language Models are Unsupervised Multitask Learners (GPT-2 Paper)](https://openai.com/research/language-unsupervised)
