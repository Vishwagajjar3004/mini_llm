"""
train.py
========
Implements STEPS 13-17:
  13. Prepare the training data
  14. Write the training loop
  15. Calculate loss
  16. Optimize the model
  17. Save model checkpoints

HOW LANGUAGE MODEL TRAINING WORKS, CONCEPTUALLY:
We take a long stream of token IDs and cut it into overlapping fixed-
length chunks. For each chunk, we ask the model to predict, at every
position, what the NEXT token is. Since we already know the real next
token (it's right there in the training text), we can measure how wrong
the model's prediction was (the "loss") and adjust the model's weights
to do slightly better next time. Repeating this thousands of times is
"training".

Example, block_size = 4:
    full token stream: [5, 12, 47, 3, 8, 19, ...]
    input  (x): [5, 12, 47, 3]
    target (y): [12, 47, 3, 8]     <- x shifted one position to the left

    At position 0: model sees [5]                -> should predict 12
    At position 1: model sees [5, 12]             -> should predict 47
    At position 2: model sees [5, 12, 47]         -> should predict 3
    At position 3: model sees [5, 12, 47, 3]      -> should predict 8
  (In practice, all 4 predictions happen in ONE forward pass, thanks to
   the causal mask described in attention.py.)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Allow `python src/train.py` to find the `src` package regardless of
# which directory it's run from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import Tokenizer
from src.model import MiniLLM


# ----------------------------------------------------------------------
# STEP 13: Prepare the training data
# ----------------------------------------------------------------------
class TextDataset(Dataset):
    """
    Turns one long stream of token IDs into many overlapping training
    examples of fixed length `block_size`. PyTorch's DataLoader will use
    this class to hand the training loop one batch at a time.
    """

    def __init__(self, token_ids: list, block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        # Number of possible starting positions for a chunk of length
        # block_size + 1 (block_size input tokens + 1 shifted target).
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input tokens
        y = torch.tensor(chunk[1:], dtype=torch.long)   # targets (shifted by 1)
        return x, y


def load_dataset_texts(path: str) -> list:
    """Reads the dataset file and returns a list of non-empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def train(
    data_path: str = "data/dataset.txt",
    checkpoint_dir: str = "checkpoints",
    block_size: int = 32,
    batch_size: int = 16,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    num_epochs: int = 60,
    learning_rate: float = 3e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Build tokenizer + vocabulary (Steps 3-5) ----
    texts = load_dataset_texts(data_path)
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Turn every line into token IDs and concatenate into one long
    # stream (this also lets the model learn how sentences relate).
    all_ids = []
    for line in texts:
        all_ids.extend(tokenizer.encode(line))

    dataset = TextDataset(all_ids, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Training examples: {len(dataset)}")

    # ---- Build the model (Steps 6-12) ----
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ---- STEP 15: Loss function ----
    # CrossEntropyLoss compares the model's predicted probability
    # distribution against the true next token, for every position at
    # once. ignore_index tells it to skip <PAD> tokens.
    pad_id = tokenizer.word2id[Tokenizer.PAD_TOKEN]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # ---- STEP 16: Optimizer ----
    # Adam adjusts each weight individually based on the gradient's
    # recent history, which tends to train faster/more reliably than
    # plain gradient descent.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---- STEP 14: The training loop ----
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # 1. Forward pass: get the model's predictions.
            logits = model(x)  # (batch, seq_len, vocab_size)

            # 2. Reshape for CrossEntropyLoss, which expects:
            #    predictions: (N, num_classes)   targets: (N,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
                y.view(-1),                          # (batch*seq_len,)
            )

            # 3. Backward pass: compute gradients of loss w.r.t. every weight.
            optimizer.zero_grad()  # clear old gradients
            loss.backward()         # backpropagation

            # 4. Clip gradients to prevent occasional huge updates that
            #    can destabilize training (common practice for Transformers).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 5. Update weights using the gradients.
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    # ---- STEP 17: Save the trained model + tokenizer ----
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "mini_llm.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "d_ff": d_ff,
                "max_seq_len": block_size,
                "dropout": dropout,
            },
        },
        checkpoint_path,
    )
    tokenizer.save(os.path.join(checkpoint_dir, "vocab.json"))
    print(f"\nSaved model to {checkpoint_path}")
    print(f"Saved vocabulary to {os.path.join(checkpoint_dir, 'vocab.json')}")


if __name__ == "__main__":
    train()
