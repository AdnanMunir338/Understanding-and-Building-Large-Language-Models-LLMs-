# """
# instruction_finetune_ddp.py — Instruction fine-tuning of GPT-2 on Alvis
# =========================================================================
# Trains GPT-2 on combined Alpaca + literary instruction-response pairs.
# Structured identically to finetune_ddp.py for easy comparison.

# Steps:
#   1. Run prepare_instruction_data.py ONCE to build instruction_data.json
#   2. Submit this script via instruction_finetune_ddp.sh

# Launch:
#     torchrun --standalone --nproc_per_node=4 instruction_finetune_ddp.py
# """

# import os
# import re
# import json
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler

# os.environ["HF_HOME"]            = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"

# from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# # ═══════════════════════════════════════════════════════════════════════════════
# #  DDP helpers  (identical to finetune_ddp.py)
# # ═══════════════════════════════════════════════════════════════════════════════

# def setup_ddp():
#     dist.init_process_group(backend="nccl")
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     return local_rank


# def cleanup_ddp():
#     dist.destroy_process_group()


# def is_main_process():
#     return dist.get_rank() == 0


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Dataset — instruction-response pairs
# # ═══════════════════════════════════════════════════════════════════════════════

# class InstructionDataset(Dataset):
#     """
#     Each sample is a fully formatted instruction-response string:

#         ### Instruction:
#         Write a passage in the style of Dickens.

#         ### Response:
#         It was a cold grey morning...<|endoftext|>

#     The model is trained to predict ALL tokens (instruction + response).
#     At inference time we feed only the instruction and let it generate the response.
#     """
#     def __init__(self, data, tokenizer, max_length):
#         self.samples = []
#         for item in data:
#             encoded = tokenizer.encode(
#                 item["text"],
#                 truncation  = True,
#                 max_length  = max_length + 1,
#             )
#             if len(encoded) < 8:    # skip degenerate samples
#                 continue
#             self.samples.append(torch.tensor(encoded, dtype=torch.long))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         tokens = self.samples[idx]
#         return tokens[:-1], tokens[1:]   # input, target (next-token prediction)


# def collate_fn(batch):
#     """Pad variable-length sequences in a batch to the same length."""
#     inputs, targets = zip(*batch)
#     max_len = max(x.size(0) for x in inputs)
#     pad_inputs  = torch.zeros(len(inputs),  max_len, dtype=torch.long)
#     pad_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)  # -100 = ignore in loss
#     for i, (inp, tgt) in enumerate(zip(inputs, targets)):
#         pad_inputs[i,  :inp.size(0)] = inp
#         pad_targets[i, :tgt.size(0)] = tgt
#     return pad_inputs, pad_targets


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Loss
# # ═══════════════════════════════════════════════════════════════════════════════

# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch  = input_batch.to(device)
#     target_batch = target_batch.to(device)
#     logits = model(input_batch).logits
#     # target uses -100 for padding → cross_entropy ignores those positions
#     return torch.nn.functional.cross_entropy(
#         logits.flatten(0, 1),
#         target_batch.flatten(),
#         ignore_index=-100,
#     )


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.0
#     if len(data_loader) == 0:
#         return float("nan")
#     n = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
#     for i, (inp, tgt) in enumerate(data_loader):
#         if i >= n:
#             break
#         total_loss += calc_loss_batch(inp, tgt, model, device).item()
#     return total_loss / n


# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
#         val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
#     model.train()
#     tl = torch.tensor(train_loss, device=device)
#     vl = torch.tensor(val_loss,   device=device)
#     dist.all_reduce(tl, op=dist.ReduceOp.AVG)
#     dist.all_reduce(vl, op=dist.ReduceOp.AVG)
#     return tl.item(), vl.item()


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Text sample — feed an instruction, print the response (rank-0 only)
# # ═══════════════════════════════════════════════════════════════════════════════

# def generate_and_print_sample(model, tokenizer, device,
#                                instruction="Write a short passage in a Victorian literary style."):
#     if not is_main_process():
#         return
#     raw_model = model.module
#     raw_model.eval()

#     prompt    = f"### Instruction:\n{instruction}\n\n### Response:\n"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     with torch.no_grad():
#         output = raw_model.generate(
#             input_ids,
#             max_new_tokens     = 100,
#             temperature        = 0.8,
#             top_k              = 40,
#             do_sample          = True,
#             repetition_penalty = 1.2,
#             pad_token_id       = tokenizer.eos_token_id,
#             eos_token_id       = tokenizer.eos_token_id,
#         )

#     # Print only the generated response (not the prompt)
#     full_text  = tokenizer.decode(output[0], skip_special_tokens=True)
#     response   = full_text.split("### Response:")[-1].strip()
#     print(f"  Instruction: {instruction}")
#     print(f"  Response   : {response[:200]}")

#     raw_model.train()


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Training loop  (identical structure to finetune_ddp.py)
# # ═══════════════════════════════════════════════════════════════════════════════

# def train_instruction(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs, eval_freq, eval_iter,
#     patience=3, min_delta=1e-4,
#     checkpoint_path="best_instruction.pth",
#     tokenizer=None,
# ):
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen       = 0
#     global_step       = -1
#     best_val_loss     = float("inf")
#     epochs_no_improve = 0
#     stop_flag         = torch.tensor(0, dtype=torch.int32, device=device)

#     for epoch in range(num_epochs):
#         model.train()

#         if hasattr(train_loader.sampler, "set_epoch"):
#             train_loader.sampler.set_epoch(epoch)

#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()
#             tokens_seen += input_batch.numel()
#             global_step += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter
#                 )
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 if is_main_process():
#                     print(
#                         f"Ep {epoch+1} (Step {global_step:06d}): "
#                         f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
#                     )

#         # Print a sample response after each epoch
#         if tokenizer:
#             generate_and_print_sample(model, tokenizer, device)

#         # ── Early stopping ──────────────────────────────────────────────────
#         if is_main_process():
#             current_val = val_losses[-1] if val_losses else float("inf")
#             if current_val < best_val_loss - min_delta:
#                 best_val_loss     = current_val
#                 epochs_no_improve = 0
#                 torch.save(model.module.state_dict(), checkpoint_path)
#                 print(f"  ✓ New best val loss {best_val_loss:.4f} — checkpoint saved.")
#             else:
#                 epochs_no_improve += 1
#                 print(
#                     f"  No improvement for {epochs_no_improve}/{patience} epoch(s). "
#                     f"Best val loss: {best_val_loss:.4f}"
#                 )
#             if epochs_no_improve >= patience:
#                 print(f"\nEarly stopping at epoch {epoch+1}.")
#                 stop_flag.fill_(1)

#         dist.broadcast(stop_flag, src=0)
#         if stop_flag.item() == 1:
#             break

#     return train_losses, val_losses, track_tokens_seen, epoch + 1


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Plotting  (identical to finetune_ddp.py)
# # ═══════════════════════════════════════════════════════════════════════════════

# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots()
#     ax1.plot(epochs_seen, train_losses,               label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("Tokens seen")
#     fig.tight_layout()


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Main
# # ═══════════════════════════════════════════════════════════════════════════════

# scheduler = None

# def main(config, settings):
#     global scheduler

#     # ── 1. DDP ────────────────────────────────────────────────────────────────
#     local_rank = setup_ddp()
#     device     = torch.device(f"cuda:{local_rank}")
#     world_size = dist.get_world_size()
#     torch.manual_seed(123)

#     # ── 2. Tokenizer ──────────────────────────────────────────────────────────
#     if is_main_process():
#         print("Loading GPT-2 tokenizer...")
#     tokenizer = GPT2TokenizerFast.from_pretrained(config["variant"])
#     tokenizer.pad_token = tokenizer.eos_token

#     # ── 3. Load instruction dataset ───────────────────────────────────────────
#     data_path = "./data/instruction_data.json"
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(
#             f"{data_path} not found. "
#             "Run prepare_instruction_data.py first."
#         )

#     if is_main_process():
#         print(f"Loading instruction data from {data_path}...")

#     with open(data_path, "r") as f:
#         all_data = json.load(f)

#     # 90/10 train/val split
#     split_idx   = int(0.9 * len(all_data))
#     train_data  = all_data[:split_idx]
#     val_data    = all_data[split_idx:]

#     if is_main_process():
#         print(f"Train pairs: {len(train_data):,}  |  Val pairs: {len(val_data):,}")

#     # ── 4. Datasets ───────────────────────────────────────────────────────────
#     train_dataset = InstructionDataset(train_data, tokenizer, config["context_length"])
#     val_dataset   = InstructionDataset(val_data,   tokenizer, config["context_length"])

#     if is_main_process():
#         print(f"Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")

#     # ── 5. DistributedSamplers + DataLoaders ─────────────────────────────────
#     train_sampler = DistributedSampler(
#         train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
#     )
#     val_sampler = DistributedSampler(
#         val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size  = settings["batch_size"],
#         sampler     = train_sampler,
#         collate_fn  = collate_fn,
#         drop_last   = True,
#         num_workers = 4,
#         pin_memory  = True,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size  = settings["batch_size"],
#         sampler     = val_sampler,
#         collate_fn  = collate_fn,
#         drop_last   = False,
#         num_workers = 4,
#         pin_memory  = True,
#     )

#     # ── 6. Load GPT-2 → DDP ───────────────────────────────────────────────────
#     if is_main_process():
#         print(f"Loading pretrained GPT-2 ({config['variant']})...")
#     model = GPT2LMHeadModel.from_pretrained(config["variant"])
#     model.resize_token_embeddings(len(tokenizer))
#     model.to(device)
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank)

#     n_params = sum(p.numel() for p in model.parameters()) / 1e6
#     if is_main_process():
#         print(f"Model loaded  ({n_params:.1f}M parameters)\n")

#     # ── 7. Optimizer + scheduler ──────────────────────────────────────────────
#     total_steps  = len(train_loader) * settings["num_epochs"]
#     warmup_steps = int(0.05 * total_steps)

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr           = settings["learning_rate"],
#         weight_decay = settings["weight_decay"],
#     )
#     scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer,
#         start_factor = 0.1,
#         end_factor   = 1.0,
#         total_iters  = warmup_steps,
#     )

#     # ── 8. Train ──────────────────────────────────────────────────────────────
#     train_losses, val_losses, tokens_seen, epochs_ran = train_instruction(
#         model, train_loader, val_loader, optimizer, device,
#         num_epochs      = settings["num_epochs"],
#         eval_freq       = 5,
#         eval_iter       = 1,
#         patience        = settings["patience"],
#         min_delta       = settings["min_delta"],
#         checkpoint_path = "best_instruction.pth",
#         tokenizer       = tokenizer,
#     )

#     # ── 9. Post-training ──────────────────────────────────────────────────────
#     if is_main_process():
#         epochs_tensor = torch.linspace(0, epochs_ran, len(train_losses))
#         plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
#         plt.savefig("instruction_loss.pdf")
#         print("Loss plot saved to instruction_loss.pdf")

#         torch.save(model.module.state_dict(), "instruction_final.pth")
#         print("Final model saved to instruction_final.pth")

#     cleanup_ddp()
#     return train_losses, val_losses, tokens_seen, model


# # ═══════════════════════════════════════════════════════════════════════════════
# #  Entry point
# # ═══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":

#     FINETUNE_CONFIG = {
#         "variant":        "gpt2",
#         "context_length": 256,
#     }

#     OTHER_SETTINGS = {
#         "learning_rate": 5e-5,
#         "num_epochs":    10,       # instruction fine-tuning needs fewer epochs
#         "batch_size":    4,
#         "weight_decay":  0.01,
#         "patience":      3,        # tighter — overfitting happens fast
#         "min_delta":     1e-4,
#     }

#     main(FINETUNE_CONFIG, OTHER_SETTINGS)

##########################################################################################
"""
instruction_finetune_ddp.py — Instruction fine-tuning of GPT-2 on Alvis
=========================================================================
Trains GPT-2 on combined Alpaca + literary instruction-response pairs.

Changes vs. original:
  - Response-only loss masking: gradients only flow through response tokens,
    preventing stylistic bleeding from the instruction prefix into outputs.
  - Label smoothing (0.1): reduces model overconfidence → less hallucination.
  - Stronger weight decay (0.1 vs 0.01).
  - Cosine LR schedule with linear warmup for smoother generalization.
  - eval_iter raised to 5 for a less noisy validation signal.
  - Inference: lower temperature (0.7) + higher repetition_penalty (1.3).

Steps:
  1. Run prepare_instruction_data.py ONCE to build instruction_data.json
  2. Submit this script via instruction_finetune_ddp.sh

Launch:
    torchrun --standalone --nproc_per_node=4 instruction_finetune_ddp.py
"""

import os
import math
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ["HF_HOME"]            = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ═══════════════════════════════════════════════════════════════════════════════
#  DDP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset — instruction-response pairs with RESPONSE-ONLY loss masking
# ═══════════════════════════════════════════════════════════════════════════════

RESPONSE_MARKER = "### Response:\n"


class InstructionDataset(Dataset):
    """
    Each sample is formatted as:

        ### Instruction:
        Write a passage in the style of Dickens.

        ### Response:
        It was a cold grey morning...<|endoftext|>

    KEY CHANGE: targets for instruction-prefix tokens are masked to -100 so the
    loss (and gradients) only flow through response tokens.  This prevents the
    model from learning to mimic instruction phrasing in its outputs
    (stylistic bleeding) and focuses capacity on generating good responses.
    """
    def __init__(self, data, tokenizer, max_length):
        self.samples = []
        for item in data:
            text = item["text"]

            # Locate where the response starts inside the full string
            marker_pos = text.find(RESPONSE_MARKER)
            if marker_pos == -1:
                # Malformed sample — skip
                continue

            prefix_text   = text[: marker_pos + len(RESPONSE_MARKER)]
            response_text = text[marker_pos + len(RESPONSE_MARKER):]

            prefix_ids   = tokenizer.encode(prefix_text)
            response_ids = tokenizer.encode(response_text)

            full_ids = prefix_ids + response_ids
            if len(full_ids) > max_length + 1:
                full_ids = full_ids[: max_length + 1]
            if len(full_ids) < 8:
                continue

            tokens    = torch.tensor(full_ids,              dtype=torch.long)
            n_prefix  = len(prefix_ids)

            # Build per-token mask: 0 for instruction prefix, 1 for response
            mask = torch.zeros(len(tokens), dtype=torch.bool)
            mask[n_prefix:] = True   # only response tokens contribute to loss

            self.samples.append((tokens, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, mask = self.samples[idx]
        inp  = tokens[:-1]
        tgt  = tokens[1:].clone()
        msk  = mask[1:]           # shift mask to align with targets
        tgt[~msk] = -100          # mask out instruction-prefix positions
        return inp, tgt


def collate_fn(batch):
    """Pad variable-length sequences in a batch to the same length."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    pad_inputs  = torch.zeros(len(inputs),  max_len, dtype=torch.long)
    pad_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        pad_inputs[i,  :inp.size(0)] = inp
        pad_targets[i, :tgt.size(0)] = tgt
    return pad_inputs, pad_targets


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss
#  KEY CHANGE: label_smoothing=0.1 reduces overconfidence → less hallucination
# ═══════════════════════════════════════════════════════════════════════════════

def calc_loss_batch(input_batch, target_batch, model, device,
                    label_smoothing: float = 0.1):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch).logits
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        ignore_index    = -100,
        label_smoothing = label_smoothing,   # ← NEW
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    n = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= n:
            break
        total_loss += calc_loss_batch(inp, tgt, model, device).item()
    return total_loss / n


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()
    tl = torch.tensor(train_loss, device=device)
    vl = torch.tensor(val_loss,   device=device)
    dist.all_reduce(tl, op=dist.ReduceOp.AVG)
    dist.all_reduce(vl, op=dist.ReduceOp.AVG)
    return tl.item(), vl.item()


# ═══════════════════════════════════════════════════════════════════════════════
#  Text sample — feed an instruction, print the response (rank-0 only)
#  KEY CHANGE: lower temperature (0.7) + higher repetition_penalty (1.3)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_and_print_sample(model, tokenizer, device,
                               instruction="Write a short passage in a Victorian literary style."):
    if not is_main_process():
        return
    raw_model = model.module
    raw_model.eval()

    prompt    = f"### Instruction:\n{instruction}\n\n### Response:\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = raw_model.generate(
            input_ids,
            max_new_tokens     = 100,
            temperature        = 0.7,     # ← lowered from 0.8; less random = less hallucination
            top_k              = 40,
            do_sample          = True,
            repetition_penalty = 1.3,     # ← raised from 1.2; suppresses repetitive loops
            pad_token_id       = tokenizer.eos_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response  = full_text.split("### Response:")[-1].strip()
    print(f"  Instruction: {instruction}")
    print(f"  Response   : {response[:200]}")

    raw_model.train()


# ═══════════════════════════════════════════════════════════════════════════════
#  Cosine LR schedule with linear warmup
#  KEY CHANGE: replaces LinearLR-only; gives smoother decay → better generalisation
# ═══════════════════════════════════════════════════════════════════════════════

def get_cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps,
                                     min_lr_ratio: float = 0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_instruction(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter,
    patience=3, min_delta=1e-4,
    checkpoint_path="best_instruction.pth",
    tokenizer=None,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen       = 0
    global_step       = -1
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    stop_flag         = torch.tensor(0, dtype=torch.int32, device=device)

    for epoch in range(num_epochs):
        model.train()

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                if is_main_process():
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}  "
                        f"LR {scheduler.get_last_lr()[0]:.2e}"
                    )

        if tokenizer:
            generate_and_print_sample(model, tokenizer, device)

        # ── Early stopping ──────────────────────────────────────────────────
        if is_main_process():
            current_val = val_losses[-1] if val_losses else float("inf")
            if current_val < best_val_loss - min_delta:
                best_val_loss     = current_val
                epochs_no_improve = 0
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"  ✓ New best val loss {best_val_loss:.4f} — checkpoint saved.")
            else:
                epochs_no_improve += 1
                print(
                    f"  No improvement for {epochs_no_improve}/{patience} epoch(s). "
                    f"Best val loss: {best_val_loss:.4f}"
                )
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                stop_flag.fill_(1)

        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    return train_losses, val_losses, track_tokens_seen, epoch + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_seen, train_losses,               label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

scheduler = None


def main(config, settings):
    global scheduler

    # ── 1. DDP ────────────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    torch.manual_seed(123)

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    if is_main_process():
        print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config["variant"])
    tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Load instruction dataset ───────────────────────────────────────────
    data_path = "./data/instruction_data.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. "
            "Run prepare_instruction_data.py first."
        )

    if is_main_process():
        print(f"Loading instruction data from {data_path}...")

    with open(data_path, "r") as f:
        all_data = json.load(f)

    split_idx  = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    val_data   = all_data[split_idx:]

    if is_main_process():
        print(f"Train pairs: {len(train_data):,}  |  Val pairs: {len(val_data):,}")

    # ── 4. Datasets ───────────────────────────────────────────────────────────
    train_dataset = InstructionDataset(train_data, tokenizer, config["context_length"])
    val_dataset   = InstructionDataset(val_data,   tokenizer, config["context_length"])

    if is_main_process():
        print(f"Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")

    # ── 5. DistributedSamplers + DataLoaders ─────────────────────────────────
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = settings["batch_size"],
        sampler     = train_sampler,
        collate_fn  = collate_fn,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = settings["batch_size"],
        sampler     = val_sampler,
        collate_fn  = collate_fn,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
    )

    # ── 6. Load GPT-2 → DDP ───────────────────────────────────────────────────
    if is_main_process():
        print(f"Loading pretrained GPT-2 ({config['variant']})...")
    model = GPT2LMHeadModel.from_pretrained(config["variant"])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main_process():
        print(f"Model loaded  ({n_params:.1f}M parameters)\n")

    # ── 7. Optimizer + cosine scheduler with warmup ───────────────────────────
    total_steps  = len(train_loader) * settings["num_epochs"]
    warmup_steps = int(0.05 * total_steps)

    # KEY CHANGE: weight_decay 0.1 (up from 0.01) for stronger regularisation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = settings["learning_rate"],
        weight_decay = settings["weight_decay"],   # see OTHER_SETTINGS below
    )

    # KEY CHANGE: cosine decay replaces LinearLR
    scheduler = get_cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # ── 8. Train ──────────────────────────────────────────────────────────────
    train_losses, val_losses, tokens_seen, epochs_ran = train_instruction(
        model, train_loader, val_loader, optimizer, device,
        num_epochs      = settings["num_epochs"],
        eval_freq       = 5,
        eval_iter       = 5,           # ← raised from 1; less noisy val signal
        patience        = settings["patience"],
        min_delta       = settings["min_delta"],
        checkpoint_path = "best_instruction_less_halluci.pth",
        tokenizer       = tokenizer,
    )

    # ── 9. Post-training ──────────────────────────────────────────────────────
    if is_main_process():
        epochs_tensor = torch.linspace(0, epochs_ran, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("instruction_loss__less_halluci.pdf")
        print("Loss plot saved to instruction_loss.pdf")

        torch.save(model.module.state_dict(), "instruction_final__less_halluci.pth")
        print("Final model saved to instruction_final.pth")

    cleanup_ddp()
    return train_losses, val_losses, tokens_seen, model


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    FINETUNE_CONFIG = {
        "variant":        "gpt2",
        "context_length": 256,
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-5,
        "num_epochs":    10,
        "batch_size":    4,
        "weight_decay":  0.1,      # ← raised from 0.01
        "patience":      3,
        "min_delta":     1e-4,
    }

    main(FINETUNE_CONFIG, OTHER_SETTINGS)