"""
attention.py
============
Implements STEP 9 (self-attention) and STEP 10 (multi-head attention).

WHAT IS SELF-ATTENTION, INTUITIVELY?
For every word, self-attention asks: "which other words in this sentence
should I pay attention to, in order to understand my own meaning here?"

Example: "The animal didn't cross the street because it was too tired."
When processing the word "it", attention lets the model look back and
figure out "it" mostly refers to "animal", not "street".

HOW IT WORKS, MATHEMATICALLY (the famous formula):

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

  - Q (Query)  : "what am I looking for?"       -- one vector per token
  - K (Key)    : "what do I contain?"           -- one vector per token
  - V (Value)  : "what information do I offer?" -- one vector per token
  - Q K^T      : dot product = how well each Query matches each Key
                 (a table of "relevance scores" between every pair of tokens)
  - / sqrt(d_k): scales scores down so softmax doesn't get too extreme
  - softmax    : turns scores into probabilities (weights that sum to 1)
  - * V        : blend the Value vectors together using those weights

ASCII picture, 4-word sentence, attention weights for the word "it":

    tokens:   The   animal   ...   it
    weights: 0.05    0.80    ...  0.02   <- "it" attends mostly to "animal"
                |       |
                v       v
         final "it" vector = mostly "animal"'s Value + a little of others

CAUSAL MASKING (important for a DECODER model):
Because we're generating text left-to-right, a token must NOT be allowed
to look at future tokens (that would be like reading the answer before
writing it). We enforce this with a "causal mask" that blocks attention
to any position that comes after the current one.

    allowed (o) / blocked (X):
    pos:  0    1    2    3
    0  [  o    X    X    X ]
    1  [  o    o    X    X ]
    2  [  o    o    o    X ]
    3  [  o    o    o    o ]
"""

import math
import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    """
    STEP 9: A single scaled dot-product self-attention head.

    This class exists purely for TEACHING PURPOSES, so you can see the
    core attention mechanism in isolation before we combine several of
    these "heads" together in MultiHeadAttention below (which is what
    the actual model uses).
    """

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        # Linear layers that project the input into Query, Key, Value
        # vectors. These weight matrices are what the model LEARNS.
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)
        self.d_head = d_head

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        causal_mask: (seq_len, seq_len) boolean, True = block this position
        """
        Q = self.query(x)  # (batch, seq_len, d_head)
        K = self.key(x)    # (batch, seq_len, d_head)
        V = self.value(x)  # (batch, seq_len, d_head)

        # Q @ K^T -> raw relevance scores between every pair of tokens.
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # Block attention to future positions by setting their score to
        # -infinity, so softmax turns them into (essentially) 0.
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Turn scores into probabilities that sum to 1 across each row.
        weights = torch.softmax(scores, dim=-1)

        # Blend the Value vectors using the attention weights.
        return weights @ V  # (batch, seq_len, d_head)


class MultiHeadAttention(nn.Module):
    """
    STEP 10: Multi-Head Attention.

    WHY MULTIPLE HEADS?
    One attention head can only learn ONE way of relating words to each
    other. Multi-head attention runs SEVERAL attention heads in parallel,
    each with its own learned Q/K/V projections, so the model can capture
    different kinds of relationships simultaneously (e.g. one head might
    learn grammar/subject-verb relationships, another might learn
    topic/meaning relationships).

    ASCII picture (num_heads = 4):

        input x
          |
          +----------+----------+----------+
          v          v          v          v
        Head 1     Head 2     Head 3     Head 4     (each is like SingleHeadAttention)
          |          |          |          |
          +----------+----------+----------+
                        |
                  concatenate outputs
                        |
                 final Linear layer (mixes heads back together)
                        |
                      output
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # size of each head's Q/K/V vectors

        # NOTE: for efficiency, we compute Q, K, V for ALL heads in one
        # big matrix multiply, then split the result into `num_heads`
        # pieces, rather than looping over separate SingleHeadAttention
        # modules. The math is identical to running several
        # SingleHeadAttention heads in parallel.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # After combining all heads' outputs, mix them back together.
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_head)"""
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)  # swap seq_len and num_heads dims

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        causal_mask: (seq_len, seq_len) boolean, True = block this position
        """
        batch_size, seq_len, d_model = x.shape

        # Project input into Q, K, V, then split into multiple heads.
        Q = self._split_heads(self.q_proj(x))  # (batch, heads, seq_len, d_head)
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        # Scaled dot-product attention, computed for ALL heads at once.
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        # causal_mask shape (seq_len, seq_len) broadcasts across
        # (batch, heads, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attn_output = weights @ V  # (batch, heads, seq_len, d_head)

        # Merge heads back together: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        return self.out_proj(attn_output)
