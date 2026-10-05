"""
transformer_block.py
=====================
Implements STEP 8: the Transformer Decoder block, combining multi-head
attention (Step 10) and feed-forward (Step 11) with two other important
ingredients: residual connections and layer normalization.

WHY RESIDUAL CONNECTIONS ("x + sublayer(x)")?
Instead of replacing x entirely, we ADD the sublayer's output to the
original input. This gives gradients a direct path backward through the
network during training, which makes deep networks much easier to train
(without this trick, deep networks tend to struggle to learn at all).

WHY LAYER NORMALIZATION?
It rescales each token's vector to have a stable mean/variance, which
keeps training numerically stable. We use the "Pre-LN" arrangement
(normalize BEFORE the sublayer, not after) because it trains more
reliably, especially for small/shallow models like ours.

ASCII picture of ONE Transformer Decoder block:

    input x
      |
      +------------------------------+
      |                               |
      v                               |
  LayerNorm                           |
      |                               |
  Multi-Head Self-Attention           | (residual / "skip" connection)
      |                               |
      +<------------------------------+
      |
      v  (x = x + attention_output)
      +------------------------------+
      |                               |
      v                               |
  LayerNorm                           |
      |                               |
  Feed-Forward                        | (residual / "skip" connection)
      |                               |
      +<------------------------------+
      |
      v  (x = x + feedforward_output)
    output
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    One decoder block. The full model (Step 8) stacks several of these
    on top of each other -- more blocks = more capacity to learn
    complex patterns (at the cost of more parameters and compute).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # --- Self-attention sub-layer, with Pre-LN + residual connection ---
        normed = self.norm1(x)
        attn_out = self.attention(normed, causal_mask)
        x = x + self.dropout1(attn_out)  # residual connection

        # --- Feed-forward sub-layer, with Pre-LN + residual connection ---
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout2(ff_out)  # residual connection

        return x
