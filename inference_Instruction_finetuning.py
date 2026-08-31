"""
Usage:
    python instruction_inference.py
    python instruction_inference.py --checkpoint best_instruction.pth
    python instruction_inference.py --instruction "Who is Sherlock Holmes?"
"""

import os
import argparse
import torch

os.environ["HF_HOME"]            = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

GPT2_VARIANT = "gpt2"

# Built-in test instructions
INSTRUCTIONS = [
    "Write a short passage in a Victorian literary style.",
    "Continue the following passage: The old man walked slowly toward the sea,",
    "Who is Sherlock Holmes?",
    "Write a dramatic opening sentence for a gothic novel.",
    "What is the mood of a story that begins with a storm at sea?",
    "Write a conversation between two characters in the style of Jane Austen.",
]


def generate_response(model, tokenizer, instruction, device,
                      max_new_tokens=150, temperature=0.8, top_k=40):
    """Feed an instruction, return only the generated response."""
    prompt    = f"### Instruction:\n{instruction}\n\n### Response:\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_k              = top_k,
            do_sample          = temperature > 0,
            repetition_penalty = 1.2,
            pad_token_id       = tokenizer.eos_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text[len(tokenizer.decode(input_ids[0])):].strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="Instruction fine-tuned GPT-2 inference")
    parser.add_argument("--checkpoint",     type=str,   default="best_instruction.pth")
    parser.add_argument("--device",         type=str,   default=None)
    parser.add_argument("--max_new_tokens", type=int,   default=150)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_k",          type=int,   default=40)
    parser.add_argument("--instruction",    type=str,   default=None,
                        help="Single custom instruction (overrides built-in list)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device : {device}")

    print(f"Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_VARIANT)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model architecture...")
    model = GPT2LMHeadModel.from_pretrained(GPT2_VARIANT)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model ready  ({n_params:.1f}M parameters)\n")

    instructions = [args.instruction] if args.instruction else INSTRUCTIONS
    sep = "─" * 65

    print(sep)
    for i, instruction in enumerate(instructions, 1):
        response = generate_response(
            model, tokenizer, instruction, device,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
        )
        print(f"\n[{i}] Instruction : {instruction}")
        print(f"    Response    : {response}")
        print(sep)


if __name__ == "__main__":
    main()