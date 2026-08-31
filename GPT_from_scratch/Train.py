# import matplotlib.pyplot as plt
# import os
# import requests
# import torch
# import tiktoken
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from Attention_choices import CausalMultiHeadAttention, SlidingWindowAttention
# from layers import LayerNorm, GELU, FeedForward
# from GPT import gpt_Model, generate_text_simple
# from Dataloader import bpe_dataloader, wordpiece_dataloader, clean_text
# from datasets import load_dataset


# def load_text_data(cache_dir="./data"):
#     """Load multiple Gutenberg books as one large text corpus."""
#     os.makedirs(cache_dir, exist_ok=True)
    
#     books = {
#         "war_and_peace":      "https://www.gutenberg.org/files/2600/2600-0.txt",
#         "pride_prejudice":    "https://www.gutenberg.org/files/1342/1342-0.txt",
#         "moby_dick":          "https://www.gutenberg.org/files/2701/2701-0.txt",
#         "tale_two_cities":    "https://www.gutenberg.org/files/98/98-0.txt",
#         "sherlock_holmes":    "https://www.gutenberg.org/files/1661/1661-0.txt",
#         "dracula":            "https://www.gutenberg.org/files/345/345-0.txt",
#         "frankenstein":       "https://www.gutenberg.org/files/84/84-0.txt",
#         "great_expectations": "https://www.gutenberg.org/files/1400/1400-0.txt",
#     }
    
#     all_text = []
#     for name, url in books.items():
#         fpath = os.path.join(cache_dir, f"{name}.txt")
#         if not os.path.exists(fpath):
#             print(f"Downloading {name}...")
#             try:
#                 r = requests.get(url, timeout=30)
#                 r.raise_for_status()
#                 with open(fpath, "w", encoding="utf-8") as f:
#                     f.write(r.text)
#                 all_text.append(r.text)
#             except Exception as e:
#                 print(f"  Skipping {name}: {e}")
#         else:
#             with open(fpath, "r", encoding="utf-8") as f:
#                 all_text.append(f.read())
    
#     combined = "\n\n".join(all_text)
#     print(f"Total corpus size: {len(combined):,} characters")
#     return combined


# def text_to_token_ids(text, tokenizer):
#     encoded = tokenizer.encode(clean_text(text))
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
#     return encoded_tensor


# def token_ids_to_text(token_ids, tokenizer):
#     flat = token_ids.squeeze(0)  # remove batch dimension
#     return tokenizer.decode(flat.tolist())


# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#     logits = model(input_batch)
#     loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
#     return loss


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches


# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
#         val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
#     model.train()
#     return train_loss, val_loss


# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     encoded = text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids = generate_text_simple(
#             model=model, idx=encoded,
#             max_new_tokens=50, context_size=context_size
#         )
#         decoded_text = token_ids_to_text(token_ids, tokenizer)
#         print(decoded_text.replace("\n", " "))  # Compact print format
#     model.train()


# def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
#                        eval_freq, eval_iter, start_context, tokenizer):
#     # Initialize lists to track losses and tokens seen
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen = 0
#     global_step = -1

#     # Main training loop
#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode

#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()  # Calculate loss gradients
#             optimizer.step()  # Update model weights using loss gradients
#             tokens_seen += input_batch.numel()
#             global_step += 1

#             # Optional evaluation step
#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

#         # Print a sample text after each epoch
#         generate_and_print_sample(
#             model, tokenizer, device, start_context
#         )

#     return train_losses, val_losses, track_tokens_seen


# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots()

#     # Plot training and validation loss against epochs
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")

#     # Create a second x-axis for tokens seen
#     ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
#     ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
#     ax2.set_xlabel("Tokens seen")

#     fig.tight_layout()  # Adjust layout to make room
#     # plt.show()


# def main(gpt_config, settings):
#     import requests, os
#     torch.manual_seed(123)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     ##############################
#     # Download data if necessary
#     ##############################

#     # file_path = "the-verdict.txt"
#     # url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

#     # if not os.path.exists(file_path):
#     #     response = requests.get(url, timeout=30)
#     #     response.raise_for_status()
#     #     text_data = response.text
#     #     with open(file_path, "w", encoding="utf-8") as file:
#     #         file.write(text_data)
#     # else:
#     #     with open(file_path, "r", encoding="utf-8") as file:
#     #         text_data = file.read()

    
#     text_data = load_text_data()
#     ##############################
#     # Initialize model
#     ##############################

#     model = gpt_Model(gpt_config)
#     model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
#     )

#     ##############################
#     # Set up dataloaders
#     ##############################

#     # Train/validation ratio
#     train_ratio = 0.90
#     split_idx = int(train_ratio * len(text_data))

#     train_loader,_ ,tokenizer1 = bpe_dataloader(
#     # train_loader,_ ,tokenizer1 = wordpiece_dataloader(
#         text_data[:split_idx],
#         batch_size=settings["batch_size"],
#         max_length=gpt_config["context_length"],
#         stride=gpt_config["context_length"],
#         drop_last=True,
#         shuffle=True,
#         num_workers=0
#     )

#     val_loader,_,tokenizer1 = bpe_dataloader(
#     # val_loader,_ ,tokenizer1 = wordpiece_dataloader(
#         text_data[split_idx:],
#         batch_size=settings["batch_size"],
#         max_length=gpt_config["context_length"],
#         stride=gpt_config["context_length"],
#         drop_last=False,
#         shuffle=False,
#         num_workers=0
#     )

#     ##############################
#     # Train model
#     ##############################

#     # tokenizer = tiktoken.get_encoding("gpt2")
#     tokenizer = tokenizer1


#     train_losses, val_losses, tokens_seen = train_model_simple(
#         model, train_loader, val_loader, optimizer, device,
#         num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
#         start_context="Every effort moves you", tokenizer=tokenizer
#     )

#     return train_losses, val_losses, tokens_seen, model


# if __name__ == "__main__":

#     # GPT_CONFIG_124M = {
#     #     "vocab_size": 50257,    # Vocabulary size
#     #     "context_length": 256,  # Shortened context length (orig: 1024)
#     #     "emb_dim": 768,         # Embedding dimension
#     #     "n_heads": 12,          # Number of attention heads
#     #     "n_layers": 12,         # Number of layers
#     #     "drop_rate": 0.1,       # Dropout rate
#     #     "qkv_bias": False       # Query-key-value bias
#     # }

#     GPT_CONFIG_124M = {
#        "vocab_size":       50257,
#         "context_length":   1024,    # ↓ back to 1024 — halves attention memory
#         "emb_dim":          1280,
#         "n_heads":          20,      # 1280/20 = 64 per head
#         "n_layers":         36,      # ↓ from 48
#         "drop_rate":        0.1,
#         "qkv_bias":         False
#     }

#     OTHER_SETTINGS = {
#         "learning_rate": 5e-4,
#         "num_epochs": 100,
#         "batch_size": 2,
#         "weight_decay": 0.1
#     }

#     ###########################
#     # Initiate training
#     ###########################

#     train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

#     ###########################
#     # After training
#     ###########################

#     # Plot results
#     epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
#     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
#     plt.savefig("loss.pdf")

#     # Save and load model
#     torch.save(model.state_dict(), "model.pth")
#     model = gpt_Model(GPT_CONFIG_124M)
#     model.load_state_dict(torch.load("model.pth", weights_only=True))

############################### Early stoping ###########3
# import matplotlib.pyplot as plt
# import os
# import requests
# import torch
# import tiktoken
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from Attention_choices import CausalMultiHeadAttention, SlidingWindowAttention
# from layers import LayerNorm, GELU, FeedForward
# from GPT import gpt_Model, generate_text_simple
# from Dataloader import bpe_dataloader, wordpiece_dataloader, clean_text
# from datasets import load_dataset


# def load_text_data(cache_dir="./data"):
#     """Load multiple Gutenberg books as one large text corpus."""
#     os.makedirs(cache_dir, exist_ok=True)

#     books = {
#         "war_and_peace":      "https://www.gutenberg.org/files/2600/2600-0.txt",
#         "pride_prejudice":    "https://www.gutenberg.org/files/1342/1342-0.txt",
#         "moby_dick":          "https://www.gutenberg.org/files/2701/2701-0.txt",
#         "tale_two_cities":    "https://www.gutenberg.org/files/98/98-0.txt",
#         "sherlock_holmes":    "https://www.gutenberg.org/files/1661/1661-0.txt",
#         "dracula":            "https://www.gutenberg.org/files/345/345-0.txt",
#         "frankenstein":       "https://www.gutenberg.org/files/84/84-0.txt",
#         "great_expectations": "https://www.gutenberg.org/files/1400/1400-0.txt",
#     }

#     all_text = []
#     for name, url in books.items():
#         fpath = os.path.join(cache_dir, f"{name}.txt")
#         if not os.path.exists(fpath):
#             print(f"Downloading {name}...")
#             try:
#                 r = requests.get(url, timeout=30)
#                 r.raise_for_status()
#                 with open(fpath, "w", encoding="utf-8") as f:
#                     f.write(r.text)
#                 all_text.append(r.text)
#             except Exception as e:
#                 print(f"  Skipping {name}: {e}")
#         else:
#             with open(fpath, "r", encoding="utf-8") as f:
#                 all_text.append(f.read())

#     combined = "\n\n".join(all_text)
#     print(f"Total corpus size: {len(combined):,} characters")
#     return combined


# def text_to_token_ids(text, tokenizer):
#     encoded = tokenizer.encode(clean_text(text))
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
#     return encoded_tensor


# def token_ids_to_text(token_ids, tokenizer):
#     flat = token_ids.squeeze(0)  # remove batch dimension
#     return tokenizer.decode(flat.tolist())


# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#     logits = model(input_batch)
#     loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
#     return loss


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches


# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
#         val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
#     model.train()
#     return train_loss, val_loss


# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     encoded = text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids = generate_text_simple(
#             model=model, idx=encoded,
#             max_new_tokens=50, context_size=context_size
#         )
#         decoded_text = token_ids_to_text(token_ids, tokenizer)
#         print(decoded_text.replace("\n", " "))  # Compact print format
#     model.train()


# def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
#                        eval_freq, eval_iter, start_context, tokenizer,
#                        patience=15, min_delta=1e-4):
#     """
#     Train model with early stopping.

#     Args:
#         patience   : stop after this many epochs with no val-loss improvement.
#         min_delta  : minimum drop in val loss to count as an improvement.
#     """
#     # Initialize lists to track losses and tokens seen
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen = 0
#     global_step = -1

#     # Early-stopping state
#     best_val_loss = float("inf")
#     epochs_no_improve = 0

#     # Main training loop
#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode

#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()        # Calculate loss gradients
#             optimizer.step()       # Update model weights using loss gradients
#             tokens_seen += input_batch.numel()
#             global_step += 1

#             # Optional evaluation step
#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

#         # Print a sample text after each epoch
#         generate_and_print_sample(model, tokenizer, device, start_context)

#         # ── Early stopping check (once per epoch, on latest val loss) ──────
#         current_val = val_losses[-1] if val_losses else float("inf")
#         if current_val < best_val_loss - min_delta:
#             best_val_loss = current_val
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             print(f"  No improvement for {epochs_no_improve}/{patience} epoch(s). "
#                   f"Best val loss: {best_val_loss:.4f}")

#         # if epochs_no_improve >= patience:
#         #     print(f"\nEarly stopping triggered at epoch {epoch+1} "
#         #           f"(no improvement for {patience} consecutive epochs).")
#         #     break
#         # ───────────────────────────────────────────────────────────────────

#     return train_losses, val_losses, track_tokens_seen


# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots()

#     # Plot training and validation loss against epochs
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")

#     # Create a second x-axis for tokens seen
#     ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
#     ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
#     ax2.set_xlabel("Tokens seen")

#     fig.tight_layout()  # Adjust layout to make room
#     # plt.show()


# def main(gpt_config, settings):
#     torch.manual_seed(123)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     ##############################
#     # Load data
#     ##############################
#     # file_path = "the-verdict.txt"
#     # url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

#     # if not os.path.exists(file_path):
#     #     response = requests.get(url, timeout=30)
#     #     response.raise_for_status()
#     #     text_data = response.text
#     #     with open(file_path, "w", encoding="utf-8") as file:
#     #         file.write(text_data)
#     # else:
#     #     with open(file_path, "r", encoding="utf-8") as file:
#     #         text_data = file.read()

    

#     text_data = load_text_data()

#     ##############################
#     # Initialize model
#     ##############################

#     model = gpt_Model(gpt_config)
#     model.to(device)
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=settings["learning_rate"],
#         weight_decay=settings["weight_decay"]
#     )

#     ##############################
#     # Set up dataloaders
#     ##############################

#     train_ratio = 0.90
#     # split_idx = int(train_ratio * len(text_data))
#     split_idx = len(text_data) - 2048 

#     train_loader, _, tokenizer1 = bpe_dataloader(
#         text_data[:split_idx],
#         batch_size=settings["batch_size"],
#         max_length=gpt_config["context_length"],
#         stride= 128, #gpt_config["context_length"],
#         drop_last=True,
#         shuffle=True,
#         num_workers=0
#     )

#     val_loader, _, tokenizer1 = bpe_dataloader(
#         text_data[split_idx:],
#         batch_size=settings["batch_size"],
#         max_length=gpt_config["context_length"],
#         stride=128, #gpt_config["context_length"],
#         drop_last=False,
#         shuffle=False,
#         num_workers=0
#     )

#     ##############################
#     # Train model
#     ##############################

#     tokenizer = tokenizer1

#     train_losses, val_losses, tokens_seen = train_model_simple(
#         model, train_loader, val_loader, optimizer, device,
#         num_epochs=settings["num_epochs"],
#         eval_freq=5,
#         eval_iter=1,
#         start_context="Every effort moves you",
#         tokenizer=tokenizer,
#         patience=settings["patience"],
#         min_delta=settings["min_delta"],
#     )

#     return train_losses, val_losses, tokens_seen, model


# if __name__ == "__main__":

#     GPT_CONFIG_124M = {
#         "vocab_size":     50257,
#         "context_length": 256,
#         "emb_dim":        1280,
#         "n_heads":        20,    # 1280/20 = 64 per head
#         "n_layers":       36,
#         "drop_rate":      0.1,
#         "qkv_bias":       False,
#     }


# #     # GPT_CONFIG_124M = {
# #     #     "vocab_size": 50257,    # Vocabulary size
# #     #     "context_length": 256,  # Shortened context length (orig: 1024)
# #     #     "emb_dim": 768,         # Embedding dimension
# #     #     "n_heads": 12,          # Number of attention heads
# #     #     "n_layers": 12,         # Number of layers
# #     #     "drop_rate": 0.1,       # Dropout rate
# #     #     "qkv_bias": False       # Query-key-value bias
# #     # }

#     OTHER_SETTINGS = {
#         "learning_rate": 5e-4,
#         "num_epochs":    100,
#         "batch_size":    2,
#         "weight_decay":  0.1,
#         "patience":      5,      # epochs without improvement before stopping
#         "min_delta":     1e-4,   # minimum val-loss drop to count as improvement
#     }

#     ###########################
#     # Initiate training
#     ###########################

#     train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

#     ###########################
#     # After training
#     ###########################

#     # Plot results
#     epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
#     plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
#     plt.savefig("loss.pdf")

#     # Save and load model
#     torch.save(model.state_dict(), "model.pth")
#     model = gpt_Model(GPT_CONFIG_124M)
#     model.load_state_dict(torch.load("model.pth", weights_only=True))

################################## DDPM ####################3

import os
import matplotlib
matplotlib.use("Agg")          # no display needed on cluster
import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── your existing modules (unchanged) ────────────────────────────────────────
from Attention_choices import CausalMultiHeadAttention, SlidingWindowAttention
from layers import LayerNorm, GELU, FeedForward
from GPT import gpt_Model, generate_text_simple
from Dataloader import bpe_dataloader, wordpiece_dataloader, clean_text
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  DDP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_ddp():
    """Initialise the process group (called once per process)."""
    dist.init_process_group(backend="nccl")   # NCCL is fastest for GPU↔GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_text_data(cache_dir="./data"):
    """Download multiple Gutenberg books and return a single text corpus."""
    os.makedirs(cache_dir, exist_ok=True)
    books = {
        "war_and_peace":      "https://www.gutenberg.org/files/2600/2600-0.txt",
        "pride_prejudice":    "https://www.gutenberg.org/files/1342/1342-0.txt",
        "moby_dick":          "https://www.gutenberg.org/files/2701/2701-0.txt",
        "tale_two_cities":    "https://www.gutenberg.org/files/98/98-0.txt",
        "sherlock_holmes":    "https://www.gutenberg.org/files/1661/1661-0.txt",
        "dracula":            "https://www.gutenberg.org/files/345/345-0.txt",
        "frankenstein":       "https://www.gutenberg.org/files/84/84-0.txt",
        "great_expectations": "https://www.gutenberg.org/files/1400/1400-0.txt",
    }
    all_text = []
    for name, url in books.items():
        fpath = os.path.join(cache_dir, f"{name}.txt")
        if not os.path.exists(fpath):
            if is_main_process():
                print(f"Downloading {name}...")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(r.text)
                all_text.append(r.text)
            except Exception as e:
                if is_main_process():
                    print(f"  Skipping {name}: {e}")
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                all_text.append(f.read())
    combined = "\n\n".join(all_text)
    if is_main_process():
        print(f"Total corpus size: {len(combined):,} characters")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  Token helpers
# ═══════════════════════════════════════════════════════════════════════════════

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(clean_text(text))
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════════

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    # DDP wraps the model; call the underlying forward the same way
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average loss over `num_batches` batches (or all if None)."""
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
    """Evaluate on both splits; returns scalars averaged across all ranks."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()

    # Reduce losses across all ranks so every process sees the same numbers
    for loss_tensor in [
        torch.tensor(train_loss, device=device),
        torch.tensor(val_loss,   device=device),
    ]:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

    # Re-read the reduced values
    tl = torch.tensor(train_loss, device=device)
    vl = torch.tensor(val_loss,   device=device)
    dist.all_reduce(tl, op=dist.ReduceOp.AVG)
    dist.all_reduce(vl, op=dist.ReduceOp.AVG)
    return tl.item(), vl.item()


# ═══════════════════════════════════════════════════════════════════════════════
#  Text sample (rank-0 only)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_and_print_sample(model, tokenizer, device, start_context):
    if not is_main_process():
        return
    raw_model = model.module           # unwrap DDP
    raw_model.eval()
    context_size = raw_model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=raw_model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    raw_model.train()


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loop with early stopping
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_ddp(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, start_context, tokenizer,
    patience=5, min_delta=1e-4, checkpoint_path="best_model.pth"
):
    """
    DDP-aware training loop with early stopping on validation loss.

    Early stopping fires when val loss does not improve by `min_delta`
    for `patience` consecutive epochs. The best checkpoint is saved to
    `checkpoint_path` (rank-0 only).
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen  = 0
    global_step  = -1

    # Early-stopping state
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    stop_flag        = torch.tensor(0, dtype=torch.int32, device=device)  # shared flag

    for epoch in range(num_epochs):
        model.train()

        # ── reshuffle each epoch for DistributedSampler ────────────────────
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
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
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

        # Print sample after each epoch (rank-0 only)
        generate_and_print_sample(model, tokenizer, device, start_context)

        # ── Early stopping (rank-0 decides, broadcasts to all ranks) ───────
        if is_main_process():
            current_val = val_losses[-1] if val_losses else float("inf")
            if current_val < best_val_loss - min_delta:
                best_val_loss     = current_val
                epochs_no_improve = 0
                # Save the best checkpoint (unwrap DDP before saving)
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"  ✓ New best val loss {best_val_loss:.4f} — checkpoint saved.")
            else:
                epochs_no_improve += 1
                print(
                    f"  No improvement for {epochs_no_improve}/{patience} epoch(s). "
                    f"Best val loss: {best_val_loss:.4f}"
                )
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping triggered at epoch {epoch+1} "
                    f"(no improvement for {patience} consecutive epochs)."
                )
                stop_flag.fill_(1)

        # Broadcast the stop decision from rank-0 to every rank
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break
        # ───────────────────────────────────────────────────────────────────

    return train_losses, val_losses, track_tokens_seen


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting (rank-0 only)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_seen, train_losses,              label="Training loss")
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

def main(gpt_config, settings):
    # ── 1. Set up DDP ─────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    torch.manual_seed(123)

    # ── 2. Load data (all ranks; files are cached after first download) ───────
    text_data  = load_text_data()
    split_idx  = len(text_data) - 2048

    # ── 3. Build dataloaders with DistributedSampler ──────────────────────────
    #
    #  bpe_dataloader returns (loader, <something>, tokenizer).
    #  The second return value may be an int (dataset length) rather than the
    #  Dataset object itself — so we extract the Dataset via loader.dataset,
    #  which is always the real Dataset regardless of what bpe_dataloader returns.
    #
    train_loader_tmp, _, tokenizer = bpe_dataloader(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=128,
        drop_last=True,
        shuffle=False,       # DistributedSampler handles shuffling
        num_workers=0,       # use 0 here; DDP workers are set below
    )
    train_dataset = train_loader_tmp.dataset   # always the real Dataset object

    val_loader_tmp, _, _ = bpe_dataloader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=128,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    val_dataset = val_loader_tmp.dataset       # always the real Dataset object

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        sampler=train_sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        sampler=val_sampler,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── 4. Model → DDP ───────────────────────────────────────────────────────
    model = gpt_Model(gpt_config).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    train_losses, val_losses, tokens_seen = train_model_ddp(
        model, train_loader, val_loader, optimizer, device,
        num_epochs     = settings["num_epochs"],
        eval_freq      = 5,
        eval_iter      = 1,
        start_context  = "Every effort moves you",
        tokenizer      = tokenizer,
        patience       = settings["patience"],
        min_delta      = settings["min_delta"],
        checkpoint_path= "best_model.pth",
    )

    # ── 6. Post-training (rank-0 only) ────────────────────────────────────────
    if is_main_process():
        epochs_tensor = torch.linspace(0, settings["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")
        print("Loss plot saved to loss.pdf")

        # Final checkpoint (last epoch weights, in addition to best_model.pth)
        torch.save(model.module.state_dict(), "model_final.pth")
        print("Final model saved to model_final.pth")

    cleanup_ddp()
    return train_losses, val_losses, tokens_seen, model


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size":     50257,
        "context_length": 256,
        "emb_dim":        1280,
        "n_heads":        20,        # 1280 / 20 = 64 per head
        "n_layers":       36,
        "drop_rate":      0.1,
        "qkv_bias":       False,
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs":    100,
        "batch_size":    2,          # per-GPU batch size
        "weight_decay":  0.1,
        "patience":      5,
        "min_delta":     1e-4,
    }

    main(GPT_CONFIG_124M, OTHER_SETTINGS)