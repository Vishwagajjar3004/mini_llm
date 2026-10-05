"""
tokenizer.py
============
This file implements STEP 3 (tokenizer), STEP 4 (vocabulary), and
STEP 5 (converting words <-> token IDs). All three are combined into
one class because they share the same state (the vocabulary).

WHAT IS A TOKENIZER?
A tokenizer's job is to break raw text into small pieces ("tokens") that
the model can work with, and to convert those pieces into integer IDs
(and back again). Real-world LLMs (GPT, Claude, Gemini) use "subword"
tokenizers (like BPE) that split text into pieces smaller than whole
words -- e.g. "unhappiness" might become ["un", "happi", "ness"]. We use
a WORD-LEVEL tokenizer here because it is much easier to understand,
even though it is less powerful and can't handle words it has never
seen (that's what the <UNK> token below is for).

Example:
    "I love pizza!" -> ["i", "love", "pizza", "!"] -> [5, 12, 47, 3]
"""

import re
import json
from collections import Counter
from typing import List, Dict


class Tokenizer:
    """
    A word-level tokenizer with a fixed vocabulary built from training text.

    Special tokens (these always exist, even before we see any data):
        <PAD> - used to pad shorter sequences so a batch has equal length
        <UNK> - stands in for any word not seen while building the vocabulary
        <BOS> - "Beginning Of Sequence" marker
        <EOS> - "End Of Sequence" marker
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        # word2id maps a token string -> integer ID (e.g. "cat" -> 17)
        self.word2id: Dict[str, int] = {}
        # id2word maps an integer ID -> token string (e.g. 17 -> "cat")
        self.id2word: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # STEP 3: Splitting raw text into token strings (before ID conversion)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_into_words(text: str) -> List[str]:
        """
        Splits a string of text into a list of lowercase word/punctuation
        tokens using a regular expression.

        We treat words (letters/digits) and punctuation marks as SEPARATE
        tokens, so "pizza!" becomes ["pizza", "!"] instead of ["pizza!"].
        This keeps the vocabulary smaller and cleaner.
        """
        text = text.lower()
        # \w+        matches one-or-more "word" characters (letters/digits/underscore)
        # [^\w\s]    matches a single punctuation character (not a word char, not whitespace)
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    # ------------------------------------------------------------------
    # STEP 4: Building the vocabulary from a list of training texts
    # ------------------------------------------------------------------
    def build_vocab(self, texts: List[str]) -> None:
        """
        Scans every text, counts how often each token appears, and
        assigns each UNIQUE token an integer ID. This complete mapping
        of token -> ID (and back) IS the model's vocabulary.
        """
        counter = Counter()
        for text in texts:
            counter.update(self._split_into_words(text))

        # Special tokens always get the first IDs (0, 1, 2, 3) so their
        # position never changes, no matter what the training data is.
        vocab_list = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        # Add every other token found in the data, most frequent first
        # (not required for correctness, just keeps things tidy/readable).
        vocab_list += [word for word, _ in counter.most_common()]

        # Build the two lookup dictionaries used everywhere else.
        self.word2id = {word: idx for idx, word in enumerate(vocab_list)}
        self.id2word = {idx: word for word, idx in self.word2id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    # ------------------------------------------------------------------
    # STEP 5: Converting words <-> token IDs
    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts a raw text string into a list of integer token IDs.
        Any word not seen during build_vocab() becomes <UNK> (unknown).
        """
        words = self._split_into_words(text)
        ids = [self.word2id.get(w, self.word2id[self.UNK_TOKEN]) for w in words]

        if add_special_tokens:
            ids = [self.word2id[self.BOS_TOKEN]] + ids + [self.word2id[self.EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a list of integer token IDs back into a human-readable
        string (the reverse of encode()).
        """
        words = []
        specials = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}
        for i in ids:
            word = self.id2word.get(i, self.UNK_TOKEN)
            if skip_special_tokens and word in specials:
                continue
            words.append(word)

        # Small formatting touch: don't put a space before punctuation,
        # so we get "the cat." instead of "the cat ." .
        text = ""
        for w in words:
            if w in {".", ",", "!", "?", ";", ":"}:
                text = text.rstrip() + w + " "
            else:
                text += w + " "
        return text.strip()

    # ------------------------------------------------------------------
    # Saving / loading the vocabulary, so the training script and the
    # generation script always agree on which word maps to which ID.
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.word2id = json.load(f)
        self.id2word = {int(idx): word for word, idx in self.word2id.items()}


# ------------------------------------------------------------------
# Quick manual test: running `python src/tokenizer.py` directly shows
# the tokenizer working on a tiny example, useful for sanity checks.
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The cat sat on the mat.",
        "The dog sat on the rug.",
    ]

    tok = Tokenizer()
    tok.build_vocab(sample_texts)

    print("Vocabulary size:", tok.vocab_size)
    print("Vocabulary:", tok.word2id)

    encoded = tok.encode("The cat sat on the rug.")
    print("\nEncoded 'The cat sat on the rug.':", encoded)

    decoded = tok.decode(encoded)
    print("Decoded back:", decoded)
