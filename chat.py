"""
chat.py
=======
STEP 19: A simple command-line chat interface for the MiniLLM.

Run with:  python chat.py

IMPORTANT HONESTY NOTE:
This is NOT a "chatbot" in the ChatGPT sense (it has no concept of
conversation turns, instructions, or safety training) -- it's simply a
loop that takes whatever text you type as a PROMPT and asks the model to
continue it, the same way pressing "Tab" for autocomplete continues a
sentence. With our tiny dataset and model size, expect simple,
sometimes repetitive or nonsensical completions -- that's normal and
expected for a from-scratch educational model!
"""

import sys
from src.generate import load_model_and_tokenizer, generate_text
from src.tokenizer import Tokenizer


BANNER = """
==================================================
 MiniLLM Chat  (educational, word-level, tiny model)
==================================================
Type a prompt and press Enter to see how the model
continues it. Type 'exit' or 'quit' to leave.

Commands:
  /temp <value>   -- change sampling temperature (default 0.8)
  /len <value>    -- change number of generated tokens (default 20)
==================================================
"""


def find_unknown_words(text: str, tokenizer: Tokenizer) -> list:
    """
    Checks which words in `text` are NOT in the model's vocabulary.

    WHY THIS MATTERS:
    This is a word-level model trained on a small, fixed vocabulary. Any
    word it never saw during training (e.g. slang, names, typos, or
    words simply missing from our tiny dataset) gets silently replaced
    with the <UNK> ("unknown") token when encoded. That produces
    confusing output like "<UNK> <UNK> love the winter." instead of a
    clear explanation. Checking for unknown words BEFORE generating lets
    us warn the user honestly about what's going on, instead of hiding
    it behind cryptic tokens.
    """
    words = tokenizer._split_into_words(text)
    unknown = [w for w in words if w not in tokenizer.word2id]
    return unknown


def main():
    print("Loading model...")
    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print(BANNER)

    temperature = 0.8
    max_new_tokens = 20

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.startswith("/temp"):
            try:
                temperature = float(user_input.split()[1])
                print(f"(temperature set to {temperature})")
            except (IndexError, ValueError):
                print("Usage: /temp 0.7")
            continue

        if user_input.startswith("/len"):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"(max_new_tokens set to {max_new_tokens})")
            except (IndexError, ValueError):
                print("Usage: /len 30")
            continue

        # Warn honestly about words the model has never seen, instead of
        # letting confusing <UNK> tokens show up unexplained.
        unknown_words = find_unknown_words(user_input, tokenizer)
        if unknown_words:
            print(
                f"(note: this tiny model has never seen the word(s) "
                f"{unknown_words} in its training data -- they'll show up "
                f"as <UNK> below. Add similar sentences to data/dataset.txt "
                f"and retrain to teach it new words.)"
            )

        response = generate_text(
            user_input,
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=10,
        )
        print(f"MiniLLM: {response}\n")


if __name__ == "__main__":
    main()
