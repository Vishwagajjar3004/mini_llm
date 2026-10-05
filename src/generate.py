"""
generate.py
===========
Loads a trained checkpoint and generates text from a prompt.
Used directly by chat.py (Step 19), and can also be run standalone for
STEP 18 / STEP 20 (testing the model with different prompts).
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import Tokenizer
from src.model import MiniLLM


def load_model_and_tokenizer(checkpoint_dir: str = "checkpoints", device=None):
    """
    STEP 17 (loading half): reconstructs the exact same model
    architecture used during training, then loads its learned weights.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(checkpoint_dir, "mini_llm.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run `python src/train.py` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Rebuild the tokenizer with the SAME vocabulary used during
    # training. This is critical: token ID 47 must mean the same word
    # at generation time as it did during training!
    tokenizer = Tokenizer()
    tokenizer.load(vocab_path)

    # Rebuild the model with the same architecture, then load its weights.
    model = MiniLLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # disables dropout, etc. -- important for inference!

    return model, tokenizer, device


def generate_text(
    prompt: str,
    model: MiniLLM,
    tokenizer: Tokenizer,
    device,
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    top_k: int = 10,
) -> str:
    """
    STEP 18: Generate text continuing from `prompt`.
    """
    # Encode the prompt WITHOUT an <EOS> token (we want the model to
    # keep going, not stop immediately), but WITH a <BOS> token so the
    # model recognizes this is the start of a sequence.
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.word2id[Tokenizer.BOS_TOKEN]] + input_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.word2id[Tokenizer.EOS_TOKEN]

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_id,
    )

    return tokenizer.decode(output_ids[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from the trained MiniLLM.")
    parser.add_argument("--prompt", type=str, default="the cat", help="Starting text")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint_dir)
    result = generate_text(
        args.prompt,
        model,
        tokenizer,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {result}")
