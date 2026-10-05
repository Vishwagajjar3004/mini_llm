"""
feedforward.py
==============
Implements STEP 11: the Feed-Forward layer used inside every Transformer
block.

WHY DO WE NEED THIS, IF WE ALREADY HAVE ATTENTION?
Attention lets tokens exchange information with each other (a "mixing"
step). The feed-forward layer processes EACH token's vector
independently, giving the model extra capacity to transform / refine
what attention gathered. Think of attention as "look around and gather
information" and feed-forward as "think about what you just gathered".

Structure: Linear -> activation (ReLU) -> Linear
It first expands the vector to a bigger size (d_ff, usually 4x d_model),
applies a non-linearity, then projects it back down to d_model.

ASCII picture:

    d_model=64 --Linear--> d_ff=256 --ReLU--> d_ff=256 --Linear--> d_model=64
      (compress)            (expand: more room to compute)          (compress back)
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    A simple 2-layer fully-connected network, applied identically and
    independently to every token position (hence "position-wise").
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # expand
        self.activation = nn.ReLU()              # non-linearity
        self.linear2 = nn.Linear(d_ff, d_model)  # compress back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: same shape
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
