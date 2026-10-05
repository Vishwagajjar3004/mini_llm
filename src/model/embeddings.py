"""
embeddings.py
=============
Implements STEP 6 (word embeddings) and STEP 7 (positional encoding).

WHY DO WE NEED EMBEDDINGS?
Token IDs (like 47 for "pizza") are just arbitrary numbers -- the model
can't tell that 47 is "closer in meaning" to 48 than to 9999. An
EMBEDDING converts each ID into a vector (a list of floating point
numbers) that the model can actually learn meaningful patterns with.
These vectors start out random and get adjusted during training so that
words used in similar contexts end up with similar vectors.

Example (with a tiny embedding size of 4; real models use 64-12000+):
    "cat"   -> [ 0.12, -0.45,  0.88,  0.03]
    "dog"   -> [ 0.15, -0.40,  0.91, -0.02]   (similar to "cat"!)
    "pizza" -> [-0.77,  0.66, -0.10,  0.55]   (very different)
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Wraps PyTorch's nn.Embedding, which is essentially a lookup table:
    a matrix of shape (vocab_size, d_model) where row `i` is the
    embedding vector for token ID `i`. This matrix is LEARNED during
    training, just like any other weight in the network.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: LongTensor of shape (batch_size, seq_len)
        returns:   FloatTensor of shape (batch_size, seq_len, d_model)
        """
        # Multiplying by sqrt(d_model) is a trick from the original
        # "Attention Is All You Need" paper that keeps embedding values
        # on a similar scale to the positional encoding values below.
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    STEP 7: Positional Encoding.

    WHY DO WE NEED THIS?
    Self-attention (Step 9) looks at all tokens at once with NO built-in
    sense of order. Without help, "dog bites man" and "man bites dog"
    would look identical to it! Positional encoding adds a unique
    "signature" vector to each position (1st word, 2nd word, ...) so the
    model can tell them apart.

    We use the SINUSOIDAL encoding from the original Transformer paper:
    fixed (not learned) sine/cosine waves of different frequencies.

    ASCII picture of the idea (each row = one position's signature):

        position 0: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (mix of slow + fast waves)
        position 1: -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
        position 2: __--__--__--__--__--__--__--__
        position 3: ....................  <- every position gets a
                                              unique combination of values
    """

    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()

        # Create an empty (max_seq_len x d_model) matrix to fill in.
        pe = torch.zeros(max_seq_len, d_model)

        # position = [[0], [1], [2], ..., [max_seq_len-1]] as a column vector.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # div_term controls the frequency of each sine/cosine wave --
        # different dimensions oscillate at different speeds.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions: cosine

        # Add a batch dimension: (1, max_seq_len, d_model) so it can be
        # broadcast-added to a batch of embeddings of any batch size.
        pe = pe.unsqueeze(0)

        # register_buffer saves `pe` together with the model but marks
        # it as NOT trainable (it's fixed math, not learned weights).
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model) token embeddings
        returns: x + positional signatures, same shape
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
