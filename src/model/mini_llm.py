"""
mini_llm.py
===========
Assembles all the pieces (embeddings, positional encoding, transformer
blocks, output layer) into the final MiniLLM model, and implements text
generation (STEP 18).

FULL ARCHITECTURE, TOP TO BOTTOM:

    token ids  (batch, seq_len)
        |
    Token Embedding             (Step 6)
        + Positional Encoding   (Step 7)
        |
    TransformerBlock x N        (Step 8, built from Steps 9-11)
        |
    Final LayerNorm
        |
    Output Linear layer         (Step 12) -> logits over the vocabulary
        |
    (softmax, at generation time, to turn logits into probabilities)
"""

import torch
import torch.nn as nn

from .embeddings import TokenEmbedding, PositionalEncoding
from .transformer_block import TransformerBlock


class MiniLLM(nn.Module):
    """
    A small GPT-style (decoder-only) Transformer language model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # STEP 6 + 7: turn token ids into meaning-carrying, position-aware vectors
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)

        # STEP 8: stack of transformer decoder blocks (the "brain")
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # STEP 12: output layer -- turns each final vector into a score
        # ("logit") for every word in the vocabulary. The word with the
        # highest logit is the model's top guess for "what comes next".
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _make_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        Builds a (seq_len, seq_len) boolean mask where True means
        "block this position" (i.e. it's in the future).
        torch.triu with diagonal=1 keeps only the values ABOVE the main
        diagonal, which is exactly the set of "future" positions.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len) integer token IDs
        returns:   (batch_size, seq_len, vocab_size) raw scores ("logits")
                   for the next token, predicted at every position
        """
        seq_len = token_ids.size(1)
        device = token_ids.device

        x = self.token_embedding(token_ids)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)      # add position info
        x = self.embed_dropout(x)

        causal_mask = self._make_causal_mask(seq_len, device)

        for block in self.blocks:
            x = block(x, causal_mask)  # (batch, seq_len, d_model)

        x = self.final_norm(x)
        logits = self.output_layer(x)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = None,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """
        STEP 18: Autoregressive text generation.

        Starting from `input_ids`, repeatedly:
          1. run the model forward
          2. look at the logits for the LAST position only
          3. turn them into a probability distribution (with temperature
             and optional top-k filtering)
          4. sample the next token
          5. append it to the sequence and repeat

        temperature: controls randomness.
            < 1.0 -> more confident / predictable (sharper distribution)
            > 1.0 -> more random / creative (flatter distribution)
        top_k: if set, only sample from the k most likely next tokens
               (prevents the model from picking very unlikely / nonsensical words)
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # The model only supports sequences up to max_seq_len, so we
            # truncate to the most recent tokens if the sequence grows too long.
            context = generated[:, -self.max_seq_len:]

            logits = self.forward(context)         # (batch, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]    # only the last position matters

            # Apply temperature: dividing logits makes the distribution
            # sharper (low temp) or flatter (high temp) before softmax.
            next_token_logits = next_token_logits / max(temperature, 1e-6)

            if top_k is not None:
                # Keep only the top_k highest-scoring tokens; set the
                # rest to -infinity so they get ~0 probability after softmax.
                values, _ = torch.topk(next_token_logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < min_value, float("-inf")
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated
