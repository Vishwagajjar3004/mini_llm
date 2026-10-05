"""
llm_engine.py
=============
This is the "styling" half of the "best of both" design -- it wraps our
existing MiniLLM (the same model built in src/model/) so it can be
trained, on the fly, on the text of whatever PDF was just uploaded.

WHY RETRAIN A NEW MODEL PER PDF?
The original MiniLLM in this project has a fixed ~200-word vocabulary
built from toy sentences about cats and pizza. Real PDFs use completely
different words. Rather than pretend a 200-word model can talk about a
legal contract or a research paper, we build a FRESH, SMALL vocabulary
directly from the uploaded document's own sentences and train a small
model on just that document. This keeps every word the model generates
grounded in vocabulary the document actually uses.

TRADE-OFF, STATED HONESTLY:
This is still a tiny model trained for a few seconds on a few hundred
sentences. It will produce grammatical-ish, document-flavored text, but
it does NOT reason about the document's meaning. The factual accuracy
of an answer comes entirely from the retriever (retriever.py); MiniLLM
here only continues text in the document's "voice".
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Make the `src` package (tokenizer + model) importable from this file,
# which lives in webapp/ one level below the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import Tokenizer
from src.model import MiniLLM


class _ChunkDataset(Dataset):
    """Same sliding-window dataset used in src/train.py."""

    def __init__(self, token_ids: list, block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class DocumentLLM:
    """
    Trains and holds a small MiniLLM instance for ONE uploaded document.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        block_size: int = 32,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.block_size = block_size
        self.dropout = dropout

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer()
        self.model = None

    def train_on_sentences(self, sentences: list, epochs: int = 30, batch_size: int = 16):
        """
        Builds a vocabulary from `sentences` and trains a fresh MiniLLM
        on them. This mirrors src/train.py's train() function, but keeps
        everything in memory (no disk checkpoint) since it's meant to
        run once per uploaded document, quickly.
        """
        self.tokenizer.build_vocab(sentences)

        all_ids = []
        for s in sentences:
            all_ids.extend(self.tokenizer.encode(s))

        # Guard against documents too short to form even one training example.
        effective_block_size = min(self.block_size, max(4, len(all_ids) - 1))

        dataset = _ChunkDataset(all_ids, effective_block_size)
        if len(dataset) == 0:
            raise ValueError("Document is too short to train on.")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = MiniLLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=effective_block_size,
            dropout=self.dropout,
        ).to(self.device)

        pad_id = self.tokenizer.word2id[Tokenizer.PAD_TOKEN]
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.model.train()
        final_loss = None
        for _ in range(epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
            final_loss = total_loss / max(1, len(dataloader))

        self.model.eval()
        return {"vocab_size": self.tokenizer.vocab_size, "final_loss": final_loss}

    def generate_from_seed(
        self,
        seed_text: str,
        max_new_tokens: int = 15,
        temperature: float = 0.8,
        top_k: int = 8,
    ) -> str:
        """
        Continues `seed_text` (typically the top retrieved sentence)
        using the document-trained MiniLLM, so the output reads in the
        document's own vocabulary/style rather than generic toy text.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train_on_sentences() first.")

        input_ids = self.tokenizer.encode(seed_text, add_special_tokens=False)
        input_ids = [self.tokenizer.word2id[Tokenizer.BOS_TOKEN]] + input_ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.word2id[Tokenizer.EOS_TOKEN]

        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )
        return self.tokenizer.decode(output_ids[0].tolist())

    # ------------------------------------------------------------------
    # Persistence: save/load this document's trained model + vocabulary
    # to/from disk, so uploads survive a server restart (see storage.py).
    # ------------------------------------------------------------------
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.tokenizer.save(os.path.join(dir_path, "vocab.json"))
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "vocab_size": self.tokenizer.vocab_size,
                    "d_model": self.d_model,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "d_ff": self.d_ff,
                    "max_seq_len": self.model.max_seq_len,
                    "dropout": self.dropout,
                },
            },
            os.path.join(dir_path, "model.pt"),
        )

    @classmethod
    def load(cls, dir_path: str) -> "DocumentLLM":
        checkpoint = torch.load(
            os.path.join(dir_path, "model.pt"), map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]

        instance = cls(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            block_size=config["max_seq_len"],
            dropout=config["dropout"],
        )
        instance.tokenizer.load(os.path.join(dir_path, "vocab.json"))
        instance.model = MiniLLM(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
        ).to(instance.device)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model.eval()
        return instance
