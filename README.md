# ⚡ Text Generation with GPT-2

> **Task-01 · Prodigy Infotech Internship**  
> Fine-tune GPT-2 on a custom dataset to generate coherent and contextually relevant text that mimics the style and structure of your training data.

---

## 📌 Overview

This project fine-tunes OpenAI's **GPT-2** transformer model on a custom text corpus. After training, the model can generate new text in the same style as the training data — whether that's stories, articles, dialogues, or any other text format.

A **desktop GUI app** is also included so you can train and generate text without using the terminal.

---

## 🗂️ Project Structure

```
gpt2_text_gen/
│
├── app.py             # 🖥️  Desktop GUI — run everything with clicks
├── train.py           # 🚀  Fine-tune GPT-2 on your dataset
├── generate.py        # ✨  Generate text from the fine-tuned model
├── prepare_data.py    # 🧹  Clean and split your corpus
├── requirements.txt   # 📦  Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | Deep learning framework |
| HuggingFace Transformers | GPT-2 model & tokenizer |
| Tkinter | Desktop GUI |

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/gpt2-text-generation.git
cd gpt2-text-generation
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```



## 🚀 Usage

### Option A — GUI App (Easiest)

```bash
python app.py
```

A desktop window opens with 3 tabs — no terminal commands needed after this.

| Tab | What it does |
|---|---|
| 1 · Prepare Data | Generate sample data or load your own `.txt` file |
| 2 · Fine-Tune | Configure and start training GPT-2 |
| 3 · Generate Text | Load the model and generate text from a prompt |



### Option B — Command Line

**Step 1 — Prepare data**
```bash
# Use built-in sample data
python prepare_data.py

# Or use your own text file
python prepare_data.py --input_file my_corpus.txt
```

**Step 2 — Fine-tune GPT-2**
```bash
python train.py \
  --train_file data/train.txt \
  --model_name gpt2 \
  --output_dir output/model \
  --epochs 3 \
  --batch_size 4
```

**Step 3 — Generate text**
```bash
python generate.py \
  --model_dir output/model \
  --prompt "The future of AI is" \
  --strategy sampling \
  --temperature 0.9 \
  --max_new_tokens 200
```



## 🎛️ Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_file` | `data/train.txt` | Path to training text file |
| `--model_name` | `gpt2` | Base model (`gpt2`, `gpt2-medium`, `gpt2-large`) |
| `--output_dir` | `output/model` | Where to save the fine-tuned model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Training batch size |
| `--block_size` | `128` | Token context window |
| `--lr` | `5e-5` | Learning rate |



## 🎲 Generation Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | `output/model` | Fine-tuned model directory |
| `--prompt` | `"Once upon a time"` | Starting text |
| `--strategy` | `sampling` | `greedy` / `beam` / `sampling` |
| `--temperature` | `0.9` | Creativity (0.1 = safe, 1.5 = wild) |
| `--top_p` | `0.95` | Nucleus sampling threshold |
| `--max_new_tokens` | `200` | How many tokens to generate |



## 📊 Decoding Strategies

| Strategy | Command | Best For |
|---|---|---|
| Greedy | `--strategy greedy` | Fast baseline |
| Beam Search | `--strategy beam` | Coherent, structured text |
| Nucleus Sampling | `--strategy sampling` | Creative, varied text ✅ |

---

## 💡 Tips

- Start with **`gpt2`** (117M params) — works on CPU, trains fast.
- Use **`gpt2-medium`** (345M) for higher quality if you have a GPU.
- Training loss below **2.5** and perplexity below **12** means good quality.
- Use **temperature 0.7–1.0** for the best creative text.
- The more domain-specific your training data, the better the style transfer.


## 📉 Sample Training Output

```
INFO | Loading GPT-2 tokenizer & model (gpt2) ...
INFO | Total tokens in dataset: 24,960
INFO | Total training examples: 195
INFO | Starting fine-tuning ...
INFO | Epoch 1/3 | Step 10/49 | Loss: 3.2401 | Perplexity: 25.53
INFO | Epoch 2/3 | Step 10/49 | Loss: 2.1400 | Perplexity: 8.50
INFO | Epoch 3 complete — Avg Loss: 1.87 | Perplexity: 6.49
INFO | Model saved to: output/model
```


## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `ImportError: cannot import name 'AdamW' from 'transformers'` | Replace with `from torch.optim import AdamW` |
| `CUDA out of memory` | Use `--batch_size 1 --block_size 64` |
| `File not found: data/train.txt` | Run `python prepare_data.py` first |
| `KeyboardInterrupt` during torch load | Torch was still loading — just run again and wait |

---

## 📚 References

1. [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
2. [Language Models are Unsupervised Multitask Learners — GPT-2 Paper](https://openai.com/research/language-unsupervised)

---

## 👩‍💻 Author

**Vabhravi Pandey**  
Prodigy Infotech Internship — Task 01

---

## 📄 License

This project is for educational purposes as part of the Prodigy Infotech internship program.
