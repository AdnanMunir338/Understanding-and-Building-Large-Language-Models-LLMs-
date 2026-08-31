# import torch
# from GPT import gpt_Model, generate_text_simple
# from Dataloader import bpe_dataloader, wordpiece_dataloader, clean_text  # Imported to get the same tokenizer configuration
# import os

# def text_to_token_ids(text, tokenizer):
#     encoded = tokenizer.encode(clean_text(text))
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
#     return encoded_tensor

# def token_ids_to_text(token_ids, tokenizer):
#     flat = token_ids.squeeze(0)  # remove batch dimension
#     return tokenizer.decode(flat.tolist())

# def run_inference(prompt, model_path, gpt_config, max_tokens=50, sample_data_path="the-verdict.txt"):
#     # 1. Setup Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 2. Re-initialize Tokenizer 
#     # (Matches your main script setup by reading from the same source file)
#     if not os.path.exists(sample_data_path):
#         raise FileNotFoundError(f"Could not find '{sample_data_path}' to construct the tokenizer mapping.")
        
#     with open(sample_data_path, "r", encoding="utf-8") as file:
#         text_data = file.read()
    
#     # We take the training split index just like the training script to align tokenizer states
#     train_ratio = 0.90
#     split_idx = int(train_ratio * len(text_data))
    
#     # Extract the tokenizer instance
#     _, _, tokenizer = bpe_dataloader(
#     # _, _, tokenizer = wordpiece_dataloader(
#         text_data[:split_idx],
#         batch_size=2,
#         max_length=gpt_config["context_length"],
#         stride=128, #gpt_config["context_length"],
#         drop_last=True,
#         shuffle=False,
#         num_workers=0
#     )

#     # 3. Initialize Model Architecture and Load Weights
#     print("Loading model architecture and weights...")
#     model = gpt_Model(gpt_config)
    
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Saved weights file '{model_path}' not found. Did you run the training script?")
        
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model.to(device)
#     model.eval()  # Crucial: Put model into evaluation mode (turns off dropout)

#     # 4. Process the Input Prompt
#     print(f"\nPrompt: {prompt}")
#     context_size = gpt_config["context_length"]
#     encoded_prompt = text_to_token_ids(prompt, tokenizer).to(device)

#     # 5. Generate Text
#     print("Generating text...")
#     with torch.no_grad():
#         token_ids = generate_text_simple(
#             model=model, 
#             idx=encoded_prompt,
#             max_new_tokens=max_tokens, 
#             context_size=context_size
#         )
    
#     # 6. Decode and Print Output
#     decoded_text = token_ids_to_text(token_ids, tokenizer)
#     print("\n--- Model Output ---")
#     print(decoded_text)
#     print("--------------------")


# if __name__ == "__main__":
#     # Must explicitly match your training configuration exactly
#     # GPT_CONFIG_124M = {
#     #     "vocab_size": 50257,    
#     #     "context_length": 256,  
#     #     "emb_dim": 768,         
#     #     "n_heads": 12,          
#     #     "n_layers": 12,         
#     #     "drop_rate": 0.1,       
#     #     "qkv_bias": False       
#     # }

#     GPT_CONFIG_124M = {
#         "vocab_size":     50257,
#         "context_length": 256,
#         "emb_dim":        1280,
#         "n_heads":        20,    # 1280/20 = 64 per head
#         "n_layers":       36,
#         "drop_rate":      0.1,
#         "qkv_bias":       False,
#     }


#     # Configuration for inference execution
#     MODEL_WEIGHTS_PATH = "./best_model.pth"
#     USER_PROMPT = "Every effort moves you closer to"
#     # USER_PROMPT = "It was a dark and cold afternoon when"
#     MAX_NEW_TOKENS = 100

#     run_inference(
#         prompt=USER_PROMPT,
#         model_path=MODEL_WEIGHTS_PATH,
#         gpt_config=GPT_CONFIG_124M,
#         max_tokens=MAX_NEW_TOKENS
#     )

######################################################
# import argparse
# import torch

# from GPT import gpt_Model, generate_text_simple
# from Dataloader import clean_text

# # ── same config used during training ─────────────────────────────────────────
# GPT_CONFIG_124M = {
#     "vocab_size":     50257,
#     "context_length": 256,
#     "emb_dim":        1280,
#     "n_heads":        20,
#     "n_layers":       36,
#     "drop_rate":      0.0,   # always 0 at inference
#     "qkv_bias":       False,
# }

# # ── prompts to generate from ──────────────────────────────────────────────────
# PROMPTS = [
#     "Every effort moves you",
#     "The old man looked at the sea and",
#     "It was the best of times, it was",
#     "In the beginning there was",
#     "She opened the door and saw",
# ]


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Generation helpers
# # ═══════════════════════════════════════════════════════════════════════════════

# def load_tokenizer():
#     """BPE tokenizer — same one used during training."""
#     import tiktoken
#     return tiktoken.get_encoding("gpt2")


# def text_to_token_ids(text, tokenizer):
#     encoded = tokenizer.encode(clean_text(text))
#     return torch.tensor(encoded).unsqueeze(0)   # (1, T)


# def token_ids_to_text(token_ids, tokenizer):
#     return tokenizer.decode(token_ids.squeeze(0).tolist())


# def generate(model, tokenizer, prompt, device,
#              max_new_tokens=100, temperature=1.0, top_k=None):
#     """
#     Generate text from a prompt with optional temperature scaling and top-k sampling.

#     Args:
#         temperature : >1 → more random, <1 → more focused, 1.0 → unscaled
#         top_k       : if set, only sample from the top-k logits (None = greedy/full)
#     """
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     idx = text_to_token_ids(prompt, tokenizer).to(device)

#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             # crop context to model's context window
#             idx_cond = idx[:, -context_size:]
#             logits = model(idx_cond)           # (1, T, vocab_size)
#             logits = logits[:, -1, :]          # last position → (1, vocab_size)

#             if top_k is not None:
#                 # zero out everything outside top-k
#                 top_values, _ = torch.topk(logits, top_k)
#                 min_top = top_values[:, -1].unsqueeze(-1)
#                 logits = logits.masked_fill(logits < min_top, float("-inf"))

#             if temperature == 0.0:
#                 # pure greedy
#                 idx_next = logits.argmax(dim=-1, keepdim=True)
#             else:
#                 probs = torch.softmax(logits / temperature, dim=-1)
#                 idx_next = torch.multinomial(probs, num_samples=1)

#             idx = torch.cat([idx, idx_next], dim=1)

#     return token_ids_to_text(idx, tokenizer)


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Main
# # ═══════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description="GPT inference")
#     parser.add_argument("--checkpoint",     type=str,   default="best_model.pth",
#                         help="Path to saved model weights (.pth)")
#     parser.add_argument("--device",         type=str,   default=None,
#                         help="Device: 'cuda', 'cpu', or 'cuda:0'. Auto-detected if not set.")
#     parser.add_argument("--max_new_tokens", type=int,   default=100,
#                         help="Number of tokens to generate per prompt")
#     parser.add_argument("--temperature",    type=float, default=0.8,
#                         help="Sampling temperature (0 = greedy)")
#     parser.add_argument("--top_k",          type=int,   default=40,
#                         help="Top-k sampling (0 = disabled)")
#     parser.add_argument("--prompts",        type=str,   nargs="+", default=None,
#                         help="Custom prompts (overrides built-in list)")
#     args = parser.parse_args()

#     # ── device ────────────────────────────────────────────────────────────────
#     if args.device:
#         device = torch.device(args.device)
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}\n")

#     # ── load model ────────────────────────────────────────────────────────────
#     print(f"Loading checkpoint: {args.checkpoint}")
#     model = gpt_Model(GPT_CONFIG_124M)
#     state = torch.load(args.checkpoint, map_location=device, weights_only=True)
#     model.load_state_dict(state)
#     model.to(device)
#     model.eval()

#     n_params = sum(p.numel() for p in model.parameters()) / 1e6
#     print(f"Model loaded  ({n_params:.1f}M parameters)\n")

#     # ── tokenizer ─────────────────────────────────────────────────────────────
#     tokenizer = load_tokenizer()

#     # ── prompts ───────────────────────────────────────────────────────────────
#     prompts = args.prompts if args.prompts else PROMPTS
#     top_k   = args.top_k if args.top_k > 0 else None

#     sep = "─" * 60
#     print(sep)
#     print(f"  max_new_tokens : {args.max_new_tokens}")
#     print(f"  temperature    : {args.temperature}")
#     print(f"  top_k          : {top_k}")
#     print(sep)

#     for i, prompt in enumerate(prompts, 1):
#         output = generate(
#             model, tokenizer, prompt, device,
#             max_new_tokens = args.max_new_tokens,
#             temperature    = args.temperature,
#             top_k          = top_k,
#         )
#         print(f"\n[Sample {i}]  Prompt: \"{prompt}\"")
#         print(f"{output}")
#         print(sep)


# if __name__ == "__main__":
#     main()

#################################### 
"""
finetune_inference.py — Generate text samples from a fine-tuned GPT-2 checkpoint
==================================================================================
Usage:
    python finetune_inference.py
    python finetune_inference.py --checkpoint best_finetune.pth
    python finetune_inference.py --checkpoint best_finetune.pth --temperature 0.9 --top_k 50
    python finetune_inference.py --prompts "The old man said" "It was a dark night"
"""

import argparse
import torch

import os

# ONLY import transformers AFTER setting the env vars above
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ── GPT-2 variant used during fine-tuning ────────────────────────────────────
GPT2_VARIANT = "gpt2"   # change to "gpt2-medium" etc. if you used a larger one

# ── Built-in prompts ─────────────────────────────────────────────────────────
PROMPTS = [
    "Every effort moves you",
    "The old man looked at the sea and",
    "It was the best of times, it was",
    "She opened the letter and read",
    "Holmes leaned back in his chair and said",
    "The storm had passed and the village",
    "In the darkness of the night he",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate(model, tokenizer, prompt, device,
             max_new_tokens=120, temperature=0.8, top_k=40, top_p=0.95):
    """
    Generate text from a prompt.

    Args:
        temperature   : 0.0 = greedy, <1 = focused, >1 = creative
        top_k         : sample only from top-k tokens (0 = disabled)
        top_p         : nucleus sampling — cumulative prob cutoff (1.0 = disabled)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens  = max_new_tokens,
            temperature     = temperature if temperature > 0 else 1.0,
            top_k           = top_k if top_k > 0 else 0,
            top_p           = top_p,
            do_sample       = temperature > 0,   # greedy if temperature == 0
            repetition_penalty = 1.2,            # discourages repeating phrases
            pad_token_id    = tokenizer.eos_token_id,
            eos_token_id    = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    prompt_len    = input_ids.shape[1]
    generated_ids = output[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fine-tuned GPT-2 inference")
    parser.add_argument("--checkpoint",       type=str,   default="best_finetune.pth",
                        help="Path to fine-tuned weights (.pth)")
    parser.add_argument("--device",           type=str,   default=None,
                        help="Device: 'cuda', 'cpu'. Auto-detected if not set.")
    parser.add_argument("--max_new_tokens",   type=int,   default=120,
                        help="Tokens to generate per prompt")
    parser.add_argument("--temperature",      type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k",            type=int,   default=40,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p",            type=float, default=0.95,
                        help="Nucleus sampling probability (1.0 = disabled)")
    parser.add_argument("--prompts",          type=str,   nargs="+", default=None,
                        help="Custom prompts (overrides built-in list)")
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")

    # ── load tokenizer ────────────────────────────────────────────────────────
    print(f"Loading tokenizer ({GPT2_VARIANT})...")
    tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_VARIANT)
    tokenizer.pad_token = tokenizer.eos_token

    # ── load model architecture + fine-tuned weights ──────────────────────────
    #
    #  We load the GPT-2 architecture first (same as during fine-tuning),
    #  then overwrite its weights with your fine-tuned checkpoint.
    #  This is equivalent to what train_ddp.py does with gpt_Model + load_state_dict.
    #
    print(f"Loading base architecture ({GPT2_VARIANT})...")
    model = GPT2LMHeadModel.from_pretrained(GPT2_VARIANT)

    print(f"Loading fine-tuned weights: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model ready  ({n_params:.1f}M parameters)\n")

    # ── run generation ────────────────────────────────────────────────────────
    prompts = args.prompts if args.prompts else PROMPTS
    sep     = "─" * 65

    print(sep)
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  max_new_tokens : {args.max_new_tokens}")
    print(f"  temperature    : {args.temperature}")
    print(f"  top_k          : {args.top_k}")
    print(f"  top_p          : {args.top_p}")
    print(sep)

    for i, prompt in enumerate(prompts, 1):
        generated = generate(
            model, tokenizer, prompt, device,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
        )
        print(f"\n[Sample {i}]")
        print(f"Prompt    : {prompt}")
        print(f"Generated : {generated}")
        print(sep)


if __name__ == "__main__":
    main()
