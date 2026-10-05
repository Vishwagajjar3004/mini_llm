# MiniLLM — A Small Educational Large Language Model, Built From Scratch

This project is a **tiny, from-scratch implementation of a GPT-style Transformer
decoder language model**, built with plain PyTorch. It is designed to teach you
how LLMs actually work under the hood — tokenization, embeddings, attention,
training, and text generation — using code you can read top to bottom in an
afternoon.

**Goal:** understanding, not performance. This is NOT meant to compete with
ChatGPT, Gemini, Claude, or Llama. See the [comparison table](#how-this-differs-from-chatgpt-gemini-claude-and-llama)
at the bottom for exactly why.

---

## 1. What is an LLM? (quick recap)

An LLM is a program that has learned to **predict the next word (token)** given
the words that came before it. Generating a whole sentence just means repeating
that prediction over and over, feeding each new word back in as input
("autoregressive generation"):

```
"The sun rises in the" → predict "east"
"The sun rises in the east" → predict "."
"The sun rises in the east." → predict <END>
```

Every part of this project — tokenizer, embeddings, attention, training loop —
exists to make that one prediction as accurate as possible.

---

## 2. Project structure

```
mini-llm/
├── data/
│   └── dataset.txt              # Small training text (Step 2)
├── src/
│   ├── tokenizer.py             # Tokenizer + vocabulary + encode/decode (Steps 3-5)
│   ├── train.py                 # Data prep, training loop, loss, optimizer, checkpoints (Steps 13-17)
│   ├── generate.py              # Load checkpoint + generate text (Step 18)
│   └── model/
│       ├── __init__.py
│       ├── embeddings.py        # Token embeddings + positional encoding (Steps 6-7)
│       ├── attention.py         # Self-attention + multi-head attention (Steps 9-10)
│       ├── feedforward.py       # Feed-forward layer (Step 11)
│       ├── transformer_block.py # One decoder block (Step 8)
│       └── mini_llm.py          # Full model assembly + generation (Steps 8, 12, 18)
├── chat.py                      # Command-line chat interface (Step 19)
├── webapp/                      # Bonus: browser Chat + PDF/ZIP/Code Q&A (see Section 6)
│   ├── app.py                   # Flask backend (upload, ask, chat, library routes)
│   ├── pdf_processor.py         # PDF/ZIP text extraction; dispatches code files
│   ├── code_analyzer.py         # Structural code parsing (ast + regex)
│   ├── retriever.py             # TF-IDF retrieval (+ keyword fallback)
│   ├── llm_engine.py            # Trains/runs/saves/loads MiniLLM per upload
│   ├── storage.py               # Persists uploads to disk across restarts
│   ├── templates/
│   │   └── index.html           # Frontend (upload + chat UI, document library)
│   ├── storage/                 # Saved uploads (model + vocab + sentences)
│   ├── uploads/                 # Temp storage during processing
│   └── requirements.txt
├── checkpoints/                 # Trained models get saved here (created by train.py)
├── requirements.txt
└── README.md
```

Every file starts with a big comment block explaining **what** it does and
**why** it exists, and every important function has inline comments too.
Read them in this order to follow the 20-step journey:

`tokenizer.py` → `model/embeddings.py` → `model/attention.py` →
`model/feedforward.py` → `model/transformer_block.py` → `model/mini_llm.py` →
`train.py` → `generate.py` → `chat.py`

### The 20-Step Educational Journey

| Step | Topic | Source File | Description |
|---|---|---|---|
| **1** | Training Corpus | `data/dataset.txt` | Curated sample sentences for next-token prediction |
| **2** | Dataset Ingestion | `src/train.py` | Reading raw text lines and filtering non-empty strings |
| **3** | Text Splitting | `src/tokenizer.py` | Regex splitting separating punctuation and words |
| **4** | Vocabulary | `src/tokenizer.py` | Word frequency counting and mapping unique tokens to IDs |
| **5** | Encode & Decode | `src/tokenizer.py` | Integer conversions with `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` |
| **6** | Token Embeddings | `src/model/embeddings.py` | Mapping discrete token IDs to dense vectors of size `d_model` |
| **7** | Positional Encoding | `src/model/embeddings.py` | Fixed sinusoidal sine/cosine wave injection for sequence order |
| **8** | Decoder Stack | `src/model/transformer_block.py` | Pre-LN blocks combining attention, FFN, and residual skip connections |
| **9** | Self-Attention | `src/model/attention.py` | Scaled dot-product attention with causal future-masking |
| **10** | Multi-Head Attention | `src/model/attention.py` | Parallel attention heads capturing diverse semantic relationships |
| **11** | Feed-Forward Layer | `src/model/feedforward.py` | Position-wise 2-layer MLP expanding to `d_ff` and compressing back |
| **12** | Logit Projection | `src/model/mini_llm.py` | Final LayerNorm + linear projection from `d_model` to `vocab_size` |
| **13** | Chunking Dataset | `src/train.py` | Sliding window sequence generation (`x` inputs shifted to `y` targets) |
| **14** | Training Loop | `src/train.py` | Iterating over batches, computing gradients, and updating parameters |
| **15** | Loss Function | `src/train.py` | CrossEntropyLoss ignoring `<PAD>` tokens across all batch positions |
| **16** | Optimizer | `src/train.py` | Adam optimizer with gradient norm clipping (`max_norm=1.0`) |
| **17** | Checkpointing | `src/train.py` & `src/generate.py` | Saving and loading weights (`mini_llm.pt`) and vocabulary (`vocab.json`) |
| **18** | Text Generation | `src/model/mini_llm.py` & `src/generate.py` | Autoregressive sampling with temperature and top-k filtering |
| **19** | Interactive Chat | `chat.py` | CLI prompt-continuation loop with `/temp` and `/len` commands |
| **20** | Web App & RAG | `webapp/app.py` | Browser UI with TF-IDF retrieval + document-grounded MiniLLM generation |

### Architecture Overview

```
Input Tokens: "the cat" -> [2, 14, 28]  (BOS, "the", "cat")
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Token Embedding (vocab_size -> d_model=64)            │
│  + Positional Encoding (Sinusoidal frequencies)        │
│  + Dropout (0.1)                                       │
└────────────────────────────────────────────────────────┘
        │
        ▼  Shape: (batch, seq_len, 64)
┌────────────────────────────────────────────────────────┐
│  Transformer Decoder Block x num_layers (2)            │
│  ├── LayerNorm(64)                                     │
│  ├── MultiHeadAttention (4 heads, causal triangular mask)│
│  ├── Residual Connection: x = x + Attention(LN(x))     │
│  ├── LayerNorm(64)                                     │
│  ├── PositionwiseFeedForward (Linear 64->256->64, ReLU)│
│  └── Residual Connection: x = x + FFN(LN(x))           │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Final LayerNorm(64)                                   │
│  Linear Output Head (d_model=64 -> vocab_size)         │
└────────────────────────────────────────────────────────┘
        │
        ▼  Shape: (batch, seq_len, vocab_size)
    Next-Token Logits -> Softmax (Temperature / Top-K) -> Predicted Token
```


---

## 3. Setup

**Requirements:** Python 3.12+ and pip.

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

That's it — PyTorch is the only third-party dependency. Everything else
(tokenizer, dataset loading) uses Python's standard library.

---

## 4. How to run the project

### Step A — Train the model

```bash
python src/train.py
```

Expected output (numbers will vary slightly run to run):

```
Using device: cpu
Vocabulary size: 156
Training examples: 775
Model parameters: 120,220
Epoch   1/60 | Loss: 4.6278
Epoch   5/60 | Loss: 1.9791
Epoch  10/60 | Loss: 1.3451
...
Epoch  60/60 | Loss: 0.2852

Saved model to checkpoints/mini_llm.pt
Saved vocabulary to checkpoints/vocab.json
```

The **loss** number tells you how "surprised" the model is by the real next
word, on average — lower is better. Watching it drop from ~4.6 to ~0.3 across
training is literally watching the model learn.

This takes well under a minute on a normal laptop CPU — no GPU required.

### Step B — Generate text from a prompt

```bash
python src/generate.py --prompt "the cat" --max_new_tokens 15
```

Expected output:

```
Prompt: the cat
Generated: the cat sleeps on the bed.
```

Try different prompts to see how the model completes them:

```bash
python src/generate.py --prompt "the sun" --max_new_tokens 15
python src/generate.py --prompt "i love" --max_new_tokens 15
python src/generate.py --prompt "the river" --temperature 1.2 --max_new_tokens 15
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--prompt` | The starting text |
| `--max_new_tokens` | How many words to generate |
| `--temperature` | Randomness: below 1.0 = safer/more repetitive, above 1.0 = more creative/random |
| `--top_k` | Only sample from the k most likely next words |

### Step C — Chat with it on the command line

```bash
python chat.py
```

```
You: the dog
MiniLLM: the dog sat on the rug.

You: /temp 0.5
(temperature set to 0.5)

You: the river
MiniLLM: the river flows to the sea.

You: exit
Goodbye!
```

Type `/temp <value>` or `/len <value>` mid-chat to adjust generation
behavior, and `exit`/`quit` to leave.

**Note:** this is autocomplete, not a real chatbot — it has no memory of
previous turns and no instruction-following ability. It simply continues
whatever text you give it, based on patterns learned from `data/dataset.txt`.

---

## 5. Customizing the project

- **Add more training data:** edit `data/dataset.txt` (one sentence per line).
  More/varied data → richer vocabulary → more interesting generations. Very
  short or repetitive datasets (like the sample provided) will make the model
  memorize sentences almost exactly — that's expected and fine for learning
  purposes.
- **Make the model bigger/smaller:** edit the arguments to `train()` at the
  bottom of `src/train.py` (`d_model`, `num_heads`, `num_layers`, `d_ff`,
  `block_size`). Bigger = more capacity but slower training.
- **Train longer:** increase `num_epochs` in `src/train.py`.

---

## 6. Bonus: browser-based Chat + PDF/ZIP/Code Q&A web app

`webapp/` contains a small Flask application with **two tabs**, so the
whole project can be used entirely from the browser without touching
the command line:

**Tab 1 — Document Q&A.** Upload a `.pdf`, a `.zip` (containing
multiple `.pdf`/`.txt` files), or a **code file** (`.py`, `.js`, `.ts`,
`.java`, `.c`, `.cpp`, `.go`, `.rb`, `.php`, and more). Every upload is
**saved to disk** (`webapp/storage/`), so it's still there — searchable,
no re-training needed — after you restart the server. A "Previously
uploaded" list lets you jump back into any past upload with one click,
or delete it.

- **For PDFs/ZIPs:** questions get answered two ways, side by side —
  **Found** (amber, TF-IDF search over the real text, factually
  grounded) and **MiniLLM says** (violet, that text continued by a
  model trained live on the upload's own vocabulary — styled, not
  guaranteed accurate). A broad question like *"summarize this"*
  returns a genuine overview (the opening sentences) instead of a
  forced keyword match. **Tables are detected and extracted
  separately** from surrounding paragraph text (using bounding-box
  detection), so a question like *"what is Q2 for North"* gets an
  exact answer instead of a garbled run-on sentence of jumbled numbers.
- **For code files (single file, or a whole ZIP of a project):** each
  file is parsed with Python's real `ast` module (or careful regex for
  other languages) to extract imports, classes, functions, and
  docstrings — 100% mechanical, not guessed. Dependency folders
  (`node_modules/`, `venv/`, `.git/`, etc.) inside a ZIP are
  automatically skipped. Ask **"explain this code"** for the full
  structural breakdown of every file, or a narrow question like *"what
  does transcribe_audio do"* to search just that part.

**Tab 2 — Chat.** Talk directly to the base MiniLLM (the same one
`chat.py` uses, trained on `data/dataset.txt`) — no upload needed.

### Why answers are split into "found" vs "styled"

A ~100K-parameter word-level model has no real reading comprehension or
code understanding — asking it to answer questions directly would
produce confident-sounding nonsense. Separating "find/parse the real
content" (retrieval or code parsing — accurate) from "make it sound
natural" (MiniLLM generation — styled) is a simplified version of the
same idea behind real-world **RAG (Retrieval-Augmented Generation)**
systems used with production LLMs.

### Running the web app

```bash
cd webapp
pip install -r requirements.txt
python app.py
```

By default, the server binds to `0.0.0.0:5000` (configurable via `HOST` and `PORT` environment variables):
* **Local browser:** [http://127.0.0.1:5000](http://127.0.0.1:5000)
* **Local network / Remote dev link:** `http://<your-machine-ip>:5000` (e.g. `http://192.168.1.13:5000`)

To change host or port:
```bash
HOST=127.0.0.1 PORT=8080 python app.py
```

### Web API Reference

The backend provides a RESTful JSON API used by the frontend:

| Method | Endpoint | Description | Request / Payload | Response |
|---|---|---|---|---|
| `GET` | `/` | Serves the main web UI | None | HTML page |
| `POST` | `/api/upload` | Uploads PDF, ZIP, or code file; analyzes, indexes, and trains model | `multipart/form-data` with `file` | `{"doc_id", "filename", "doc_type", "summary", "num_sentences"}` |
| `POST` | `/api/ask` | Queries the document using TF-IDF retrieval + MiniLLM styling | `{"doc_id": "...", "question": "..."}` | `{"found": "...", "styled": "...", "score": 0.85, ...}` |
| `POST` | `/api/chat` | Direct autocomplete chat with base model (no upload required) | `{"message": "the cat"}` | `{"response": "the cat sleeps...", "unknown_words": [...]}` |
| `GET` | `/api/documents` | Lists all previously uploaded documents | None | `{"documents": [{"doc_id", "filename", ...}]}` |
| `POST` | `/api/select` | Loads a stored document from disk into active session cache | `{"doc_id": "..."}` | `{"status": "ok", "filename": "...", ...}` |
| `DELETE` | `/api/documents/<id>` | Deletes a stored document and model checkpoint from disk | None | `{"status": "deleted"}` |

**What happens on upload:** text/structure is extracted, indexed for
search (`scikit-learn` TF-IDF), and a brand-new MiniLLM is trained from
scratch on it (a few seconds). The trained model, vocabulary, and
source sentences are then saved under `webapp/storage/<doc_id>/` —
that's what makes it persistent.

**Limitations to know:**
- Scanned/image-only PDFs won't work — there's no OCR step.
- Inside a ZIP, `.pdf`, `.txt`, and common code file extensions are
  read; dependency/build folders (`node_modules/`, `venv/`, `.git/`,
  `dist/`, etc.) are skipped automatically, and code-file parsing is
  capped at 60 files per ZIP to keep uploads fast.
- Code parsing is fully accurate for Python (via `ast`); other
  languages use regex-based heuristics, which are reliable for typical
  code but can miss unusual syntax.
- Table detection in PDFs works well for standard grid-style tables;
  unusually complex or borderless table layouts may not be detected as
  a table and will fall back to (occasionally jumbled) plain text.
- Long uploads are capped to the first 400 sentences for MiniLLM
  training (kept fast), though search covers everything.
- Storage is plain files on disk (no database) — fine for personal,
  local use; not built for multi-user production deployment.
- This is a local development server (Flask's `debug=False` built-in
  server), not meant for production/public deployment.


---

## 7. Common errors and how to fix them

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch isn't installed in your active environment | Run `pip install -r requirements.txt` (make sure your venv is activated) |
| `FileNotFoundError: No checkpoint found at checkpoints/mini_llm.pt` | You ran `generate.py` or `chat.py` before training | Run `python src/train.py` first |
| `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)` | Usually means the checkpoint's vocabulary/config doesn't match the model code being used to load it | Retrain from scratch after changing model code: delete `checkpoints/` and rerun `train.py` |
| Loss stays high (>3) and doesn't drop | Learning rate too high/low, or too few epochs | Try more `num_epochs`, or tweak `learning_rate` in `train.py` |
| Generated text is repetitive/nonsensical | Normal for a tiny model trained on a tiny dataset! | Add more training data, train longer, or lower `temperature`/`top_k` for safer output |
| `ModuleNotFoundError: No module named 'src'` | Running a script from the wrong directory | Always run commands from the `mini-llm/` project root, e.g. `python src/train.py`, not from inside `src/` |
| Training is very slow | Running on CPU with a large model/dataset | Reduce `d_model`/`num_layers`, or use a machine with a CUDA GPU (the code automatically uses `cuda` if available) |
| `CUDA out of memory` | Batch size or model too large for your GPU | Lower `batch_size` or `d_model` in `train.py` |

---

## 8. How this differs from ChatGPT, Gemini, Claude, and Llama

This project uses the **same fundamental architecture** (Transformer decoder,
self-attention, embeddings, autoregressive generation) as every major LLM
today. The differences are entirely of **scale, training, and engineering**,
not of core design:

| Aspect | MiniLLM (this project) | ChatGPT / Gemini / Claude / Llama |
|---|---|---|
| Parameters | ~100K–5M | Tens to hundreds of billions |
| Training data | A few dozen hand-written sentences | Trillions of tokens: books, websites, code, conversations |
| Tokenizer | Simple word-level (whole words only) | Subword (BPE/SentencePiece-style), handles any text/language efficiently |
| Training hardware | Your laptop CPU, under a minute | Thousands of GPUs/TPUs, running for weeks to months |
| Context length | ~32 tokens | Tens of thousands to millions of tokens |
| Post-training | None — just raw next-word prediction | Extensive instruction-tuning and RLHF/RLAIF (human/AI feedback) to follow instructions, refuse harmful requests, and hold conversations |
| Knows facts/reasoning? | No — it only learned patterns in a tiny toy dataset | Yes — broad world knowledge and reasoning ability from massive, diverse training |
| Conversational memory | None (each prompt is independent) | Maintains context across a conversation |
| Safety training | None | Extensive red-teaming, safety fine-tuning, content filtering |
| Multimodality | Text only | Many support images, audio, documents, etc. |

**The big idea:** every one of those production models started from the exact
same core ingredients you just built — token embeddings, positional encoding,
multi-head self-attention, feed-forward layers, and next-token prediction.
What turns this recipe into a ChatGPT-level assistant is **scale** (way more
parameters, data, and compute) plus **extensive additional training** on top
of the base language-modeling objective (instruction tuning, human feedback,
safety alignment) — not a fundamentally different architecture.

Understanding this small model well is a genuinely good foundation for
understanding how the large ones work.

---

## 9. Suggested next experiments

- Swap the sinusoidal positional encoding for a *learned* positional
  embedding (`nn.Embedding(max_seq_len, d_model)`) and compare results.
- Add more transformer blocks (`num_layers`) and see how the loss curve changes.
- Try a character-level tokenizer instead of word-level, and compare
  vocabulary size and generation quality.
- Plot the loss curve over epochs using `matplotlib`.
- Print attention weights from `MultiHeadAttention` to visualize which words
  the model is "looking at".