"""
retriever.py
============
This is the "accurate" half of the "best of both" design.

WHY NOT JUST ASK THE TINY LLM TO ANSWER DIRECTLY?
MiniLLM is a from-scratch educational model with a tiny vocabulary and
no real reading-comprehension ability -- it can't reliably "understand"
a PDF and answer questions about it. What it IS good at is producing
natural-sounding word sequences once trained on a body of text.

So instead, we split the real answering work in two:
  1. RETRIEVAL (this file): find which sentences in the PDF are most
     relevant to the user's question, using classical NLP (TF-IDF +
     cosine similarity). This is the same technique used in the earlier
     Document Analyser project -- no neural network involved, and it's
     factually grounded in the actual PDF text.
  2. GENERATION (llm_engine.py): hand the most relevant sentence to
     MiniLLM as a starting point, and let it continue/style the text
     using patterns it learned from the document.

HOW TF-IDF + COSINE SIMILARITY WORKS, BRIEFLY:
- TF-IDF turns each sentence into a vector of numbers, where each
  number represents how important a particular word is to that
  sentence (common words like "the" score low, distinctive words score
  high).
- Cosine similarity measures the angle between two vectors -- a small
  angle (close to 1.0) means the sentence and the question use similar,
  distinctive words.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class Retriever:
    """
    Wraps a TF-IDF index over a list of sentences, so we can find the
    sentences most relevant to any question.
    """

    def __init__(self, sentences: list):
        self.sentences = sentences
        # stop_words="english" removes common filler words ("the", "is",
        # "and", ...) so the comparison focuses on meaningful content words.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # Build the TF-IDF matrix: one row per sentence, one column per
        # unique word across all sentences.
        self.sentence_vectors = self.vectorizer.fit_transform(sentences)

    def query(self, question: str, top_k: int = 3) -> list:
        """
        Returns the `top_k` sentences most relevant to `question`, each
        paired with a similarity score between 0 (unrelated) and 1
        (identical wording).

        Falls back to simple keyword-overlap matching if TF-IDF finds
        nothing (this happens for short/unusual questions where every
        word is either a stopword or missing from the document's
        vocabulary -- TF-IDF alone would report zero relevant sentences
        even though a plain keyword search might still find something
        useful).
        """
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.sentence_vectors)[0]

        top_indices = scores.argsort()[::-1][:top_k]
        results = [
            {"sentence": self.sentences[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

        if results:
            return results

        return self._keyword_fallback(question, top_k)

    def _keyword_fallback(self, question: str, top_k: int) -> list:
        """
        Plain word-overlap search: counts how many of the question's
        words appear in each sentence. Much cruder than TF-IDF, but
        catches cases TF-IDF's stricter weighting misses entirely.
        """
        question_words = set(re.findall(r"\w+", question.lower()))
        if not question_words:
            return []

        scored = []
        for sentence in self.sentences:
            sentence_words = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(question_words & sentence_words)
            if overlap > 0:
                # Normalize by question length so the "score" stays in a
                # similar 0-1 range to the TF-IDF path above.
                scored.append((overlap / len(question_words), sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"sentence": s, "score": sc} for sc, s in scored[:top_k]]
