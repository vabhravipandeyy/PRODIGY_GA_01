"""
Task-01: GPT-2 Text Generation — GUI App
Run with:  python app.py
"""

import os
import sys
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path

BG      = "#0f0f1a"
CARD    = "#1a1a2e"
ACCENT  = "#7c3aed"
ACCENT2 = "#a78bfa"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
SUCCESS = "#10b981"
ERROR   = "#ef4444"
FONT    = ("Segoe UI", 10)
FONT_B  = ("Segoe UI", 10, "bold")
MONO    = ("Consolas", 10)


class GPT2App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPT-2 Text Generation  |  Task-01")
        self.configure(bg=BG)
        self.geometry("880x720")
        self.resizable(True, True)

        self.model      = None
        self.tokenizer  = None
        self.model_dir  = tk.StringVar(value="output/model")
        self.train_file = tk.StringVar(value="data/train.txt")
        self.model_name = tk.StringVar(value="gpt2")
        self.epochs     = tk.IntVar(value=3)
        self.batch_size = tk.IntVar(value=4)
        self.prompt     = tk.StringVar(value="The future of AI is")
        self.strategy   = tk.StringVar(value="sampling")
        self.temperature= tk.DoubleVar(value=0.9)
        self.top_p      = tk.DoubleVar(value=0.95)
        self.max_tokens = tk.IntVar(value=200)

        self._build_ui()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _log(self, msg, color=TEXT):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BG, pady=6)
        outer.pack(fill="x", padx=18, pady=4)
        tk.Label(outer, text=title, font=FONT_B, bg=BG, fg=ACCENT2).pack(anchor="w", pady=(0,4))
        inner = tk.Frame(outer, bg=CARD, padx=16, pady=12)
        inner.pack(fill="x")
        return inner

    def _row(self, parent, label, widget_fn):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=FONT, bg=CARD, fg=TEXT, width=16, anchor="w").pack(side="left")
        widget_fn(row)

    def _entry(self, parent, var, width=None):
        kw = {"width": width} if width else {}
        e = tk.Entry(parent, textvariable=var, bg="#090912", fg=TEXT,
                     insertbackground=TEXT, relief="flat", font=FONT,
                     highlightthickness=1, highlightbackground=MUTED,
                     highlightcolor=ACCENT, **kw)
        e.pack(side="left", fill="x", expand=True, padx=(0,6))
        return e

    def _spin(self, parent, var, from_, to, inc=1):
        s = tk.Spinbox(parent, textvariable=var, from_=from_, to=to, increment=inc,
                       bg="#090912", fg=TEXT, buttonbackground=CARD,
                       relief="flat", font=FONT, width=8)
        s.pack(side="left")

    def _btn(self, parent, text, cmd, color=ACCENT, side="left"):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg="white", font=FONT_B,
                      relief="flat", padx=14, pady=6,
                      activebackground=ACCENT2, cursor="hand2")
        b.pack(side=side, padx=4, pady=4)
        return b

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        tk.Frame(self, bg=ACCENT, height=4).pack(fill="x")
        hdr = tk.Frame(self, bg=BG, pady=12)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="⚡  GPT-2 Text Generation", font=("Segoe UI", 17, "bold"),
                 bg=BG, fg=ACCENT2).pack(side="left")
        tk.Label(hdr, text="Task-01 · Prodigy Infotech", font=FONT, bg=BG, fg=MUTED).pack(side="left", padx=14)

        # notebook
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",      background=BG,   borderwidth=0)
        style.configure("TNotebook.Tab",  background=CARD, foreground=MUTED, font=FONT_B, padding=[14,7])
        style.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected","white")])
        style.configure("TFrame", background=BG)
        style.configure("Horizontal.TProgressbar", troughcolor=CARD, background=ACCENT)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=14, pady=(0,6))

        self._tab_data(nb)
        self._tab_train(nb)
        self._tab_generate(nb)

        # log
        lf = tk.Frame(self, bg=CARD, pady=6)
        lf.pack(fill="x", padx=14, pady=(0,12))
        tk.Label(lf, text="📋  Log", font=FONT_B, bg=CARD, fg=ACCENT2).pack(anchor="w", padx=10)
        self.log_box = scrolledtext.ScrolledText(lf, height=6, bg="#060610", fg=TEXT,
                                                  font=MONO, relief="flat", bd=0, wrap="word",
                                                  insertbackground=TEXT)
        self.log_box.pack(fill="x", padx=10, pady=(2,6))
        self.log_box.config(state="disabled")
        self._log("Ready. Follow the tabs: 1 → 2 → 3")

    # ── Tab 1: Data ───────────────────────────────────────────────────────────
    def _tab_data(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  1 · Prepare Data  ")

        c = self._card(tab, "Training Data")
        self._row(c, "Train file", lambda p: [self._entry(p, self.train_file),
                  self._btn(p, "Browse", self._browse_train, color="#1e40af")])

        btnrow = tk.Frame(tab, bg=BG)
        btnrow.pack(pady=10)
        self._btn(btnrow, "⚡ Generate Sample Data", self._prepare_sample)
        self._btn(btnrow, "📂 Use My Text File",     self._prepare_custom, color="#065f46")

        info = tk.Frame(tab, bg="#1a1040", padx=14, pady=10)
        info.pack(fill="x", padx=18, pady=4)
        tk.Label(info, bg="#1a1040", fg=ACCENT2, font=FONT, justify="left",
                 text="💡  New here? Click 'Generate Sample Data' — it creates demo training data instantly.\n"
                      "    Have your own text? Browse to your .txt file then click 'Use My Text File'.").pack(anchor="w")

    def _browse_train(self):
        p = filedialog.askopenfilename(filetypes=[("Text files","*.txt"),("All","*.*")])
        if p:
            self.train_file.set(p)

    def _prepare_sample(self):
        sample = (
            "Artificial intelligence is transforming the world.\n"
            "The transformer architecture revolutionised NLP in 2017.\n"
            "Fine-tuning adapts a pre-trained model to a new domain.\n"
            "Language models predict the next token given prior context.\n"
            "GPT-2 generates coherent long-form text from a short prompt.\n"
        ) * 80
        Path("data").mkdir(exist_ok=True)
        lines = sample.strip().split("\n")
        split = int(len(lines) * 0.9)
        Path("data/train.txt").write_text("\n".join(lines[:split]), encoding="utf-8")
        Path("data/val.txt").write_text("\n".join(lines[split:]), encoding="utf-8")
        self.train_file.set("data/train.txt")
        self._log("✅ Sample data created → data/train.txt")

    def _prepare_custom(self):
        p = self.train_file.get()
        if not os.path.isfile(p):
            messagebox.showerror("File not found", f"Cannot find:\n{p}\n\nPlease browse to your .txt file first.")
            return
        text = Path(p).read_text(encoding="utf-8")
        lines = [l for l in text.split("\n") if l.strip()]
        split = int(len(lines) * 0.9)
        Path("data").mkdir(exist_ok=True)
        Path("data/train.txt").write_text("\n".join(lines[:split]), encoding="utf-8")
        Path("data/val.txt").write_text("\n".join(lines[split:]), encoding="utf-8")
        self.train_file.set("data/train.txt")
        self._log(f"✅ Your file processed → {len(lines)} lines, saved to data/train.txt")

    # ── Tab 2: Train ──────────────────────────────────────────────────────────
    def _tab_train(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  2 · Fine-Tune  ")

        c = self._card(tab, "Training Settings")

        # model dropdown
        self._row(c, "Base model", lambda p: ttk.Combobox(p, textvariable=self.model_name,
                  values=["gpt2","gpt2-medium","gpt2-large"],
                  state="readonly", width=18, font=FONT).pack(side="left"))
        self._row(c, "Train file", lambda p: self._entry(p, self.train_file))
        self._row(c, "Output dir", lambda p: self._entry(p, self.model_dir))
        self._row(c, "Epochs",     lambda p: self._spin(p, self.epochs,     1, 20))
        self._row(c, "Batch size", lambda p: self._spin(p, self.batch_size, 1, 32))

        self.progress = ttk.Progressbar(tab, mode="indeterminate", length=500,
                                        style="Horizontal.TProgressbar")
        self.progress.pack(pady=14)
        self.train_status = tk.Label(tab, text="", font=FONT, bg=BG, fg=MUTED)
        self.train_status.pack()

        btnrow = tk.Frame(tab, bg=BG)
        btnrow.pack(pady=6)
        self._btn(btnrow, "🚀 Start Fine-Tuning", self._start_training)

    def _start_training(self):
        if not os.path.isfile(self.train_file.get()):
            messagebox.showerror("No data", "Run Step 1 first to prepare training data.")
            return
        self.progress.start(12)
        self.train_status.config(text="Training in progress…", fg=ACCENT2)
        self._log("🚀 Training started — this may take a few minutes…")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
            from torch.optim import AdamW
            from torch.utils.data import Dataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._log(f"   Device: {device}")

            tokenizer = GPT2Tokenizer.from_pretrained(self.model_name.get())
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained(self.model_name.get())
            model.to(device)

            text = Path(self.train_file.get()).read_text(encoding="utf-8")
            tokens = tokenizer.encode(text, add_special_tokens=True)
            block = 128
            examples = [torch.tensor(tokens[i:i+block], dtype=torch.long)
                        for i in range(0, len(tokens)-block+1, block)]
            self._log(f"   Examples: {len(examples)}")

            loader = DataLoader(examples, batch_size=self.batch_size.get(), shuffle=True)
            total  = len(loader) * self.epochs.get()
            opt    = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            sched  = get_linear_schedule_with_warmup(opt, max(1, total//10), total)

            model.train()
            for ep in range(1, self.epochs.get()+1):
                ep_loss = 0.0
                for step, batch in enumerate(loader, 1):
                    batch = batch.to(device)
                    loss  = model(input_ids=batch, labels=batch).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
                    ep_loss += loss.item()
                avg = ep_loss / len(loader)
                ppl = math.exp(min(avg, 20))
                self._log(f"   Epoch {ep}/{self.epochs.get()} — Loss: {avg:.4f} | Perplexity: {ppl:.2f}")

            out = Path(self.model_dir.get())
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            self._log(f"✅ Model saved → {out}")
            self.after(0, lambda: self.train_status.config(text="✅ Training complete!", fg=SUCCESS))
        except Exception as e:
            self._log(f"❌ Error: {e}", ERROR)
            self.after(0, lambda: self.train_status.config(text="❌ Training failed", fg=ERROR))
        finally:
            self.after(0, self.progress.stop)

    # ── Tab 3: Generate ───────────────────────────────────────────────────────
    def _tab_generate(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text="  3 · Generate Text  ")

        c = self._card(tab, "Generation Settings")
        self._row(c, "Model dir",    lambda p: [self._entry(p, self.model_dir),
                  self._btn(p, "Browse", self._browse_model, color="#1e40af")])
        self._row(c, "Prompt",       lambda p: self._entry(p, self.prompt))
        self._row(c, "Strategy",     lambda p: ttk.Combobox(p, textvariable=self.strategy,
                  values=["sampling","beam","greedy"], state="readonly", width=14, font=FONT).pack(side="left"))
        self._row(c, "Temperature",  lambda p: self._spin(p, self.temperature, 0.1, 2.0, 0.1))
        self._row(c, "Top-P",        lambda p: self._spin(p, self.top_p,       0.1, 1.0, 0.05))
        self._row(c, "Max tokens",   lambda p: self._spin(p, self.max_tokens,  50,  1000))

        btnrow = tk.Frame(tab, bg=BG)
        btnrow.pack(pady=6)
        self._btn(btnrow, "📦 Load Model", self._load_model, color="#065f46")
        self._btn(btnrow, "✨ Generate",   self._generate)
        self._btn(btnrow, "💾 Save Output",self._save_output, color="#92400e")

        out_lf = tk.Frame(tab, bg=CARD, padx=12, pady=8)
        out_lf.pack(fill="both", expand=True, padx=18, pady=6)
        tk.Label(out_lf, text="Generated Text", font=FONT_B, bg=CARD, fg=ACCENT2).pack(anchor="w")
        self.output_box = scrolledtext.ScrolledText(out_lf, height=10, bg="#060610", fg=TEXT,
                                                     font=MONO, relief="flat", bd=0, wrap="word",
                                                     insertbackground=TEXT)
        self.output_box.pack(fill="both", expand=True, pady=(4,0))

    def _browse_model(self):
        p = filedialog.askdirectory()
        if p:
            self.model_dir.set(p)

    def _load_model(self):
        d = self.model_dir.get()
        if not os.path.isdir(d):
            messagebox.showerror("Not found", f"Model directory not found:\n{d}\n\nComplete Step 2 first.")
            return
        self._log(f"Loading model from {d} …")
        def _load():
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained(d)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2LMHeadModel.from_pretrained(d)
                self.model.eval()
                self._log("✅ Model loaded and ready!")
            except Exception as e:
                self._log(f"❌ {e}")
        threading.Thread(target=_load, daemon=True).start()

    def _generate(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("No model", "Load a model first (click 'Load Model').")
            return
        self._log(f"Generating… prompt='{self.prompt.get()}'")
        def _gen():
            try:
                import torch
                ids = self.tokenizer.encode(self.prompt.get(), return_tensors="pt")
                with torch.no_grad():
                    s = self.strategy.get()
                    if s == "greedy":
                        out = self.model.generate(ids, max_new_tokens=self.max_tokens.get(),
                                                  pad_token_id=self.tokenizer.eos_token_id)
                    elif s == "beam":
                        out = self.model.generate(ids, max_new_tokens=self.max_tokens.get(),
                                                  num_beams=5, early_stopping=True,
                                                  pad_token_id=self.tokenizer.eos_token_id)
                    else:
                        out = self.model.generate(ids, max_new_tokens=self.max_tokens.get(),
                                                  do_sample=True,
                                                  temperature=self.temperature.get(),
                                                  top_p=self.top_p.get(),
                                                  pad_token_id=self.tokenizer.eos_token_id)
                result = self.tokenizer.decode(out[0], skip_special_tokens=True)
                self.output_box.delete("1.0", "end")
                self.output_box.insert("end", result)
                self._log("✅ Text generated!")
            except Exception as e:
                self._log(f"❌ {e}")
        threading.Thread(target=_gen, daemon=True).start()

    def _save_output(self):
        text = self.output_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty", "Nothing to save yet. Generate text first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Text","*.txt"),("All","*.*")])
        if p:
            Path(p).write_text(text, encoding="utf-8")
            self._log(f"💾 Saved to {p}")


if __name__ == "__main__":
    app = GPT2App()
    app.mainloop()