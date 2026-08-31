import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2TokenizerFast



def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0




def load_text_data(cache_dir="./data"):
    """Download 8 Gutenberg books and return a single cleaned text corpus."""
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
                print(f"  Downloading {name}...")
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
        print(f"Raw corpus size : {len(combined):,} characters")
    return combined


def clean_gutenberg(text):
   
    # 1. Remove header
    start_match = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]

    # 2. Remove footer
    end_match = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]

    # 3. Remove all-caps chapter headings (e.g. "CHAPTER I", "PART TWO")
    text = re.sub(r"^[A-Z][A-Z\s\.\-]{5,}$", "", text, flags=re.MULTILINE)

    # 4. Collapse 3+ blank lines → 1 blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip
    return text.strip()




class GPT2TextDataset(Dataset):
   
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)   # plain list of ints

        for i in range(0, len(token_ids) - max_length, stride):
            chunk  = token_ids[i : i + max_length + 1]
            self.input_ids.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.target_ids.append(torch.tensor(chunk[1:],  dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]




def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch).logits         # HuggingFace returns a dataclass
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
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




def generate_and_print_sample(model, tokenizer, device, start_context,
                               max_new_tokens=100, temperature=0.8, top_k=40):
    if not is_main_process():
        return
    raw_model = model.module
    raw_model.eval()
    input_ids = tokenizer.encode(start_context, return_tensors="pt").to(device)
    with torch.no_grad():
        output = raw_model.generate(
            input_ids,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True).replace("\n", " "))
    raw_model.train()


def finetune_model_ddp(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, start_context, tokenizer,
    patience=5, min_delta=1e-4, checkpoint_path="best_finetune.pth"
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
            # gradient clipping — important for fine-tuning stability
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
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

        generate_and_print_sample(model, tokenizer, device, start_context)

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
                print(
                    f"\nEarly stopping triggered at epoch {epoch+1} "
                    f"(no improvement for {patience} consecutive epochs)."
                )
                stop_flag.fill_(1)

        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    return train_losses, val_losses, track_tokens_seen, epoch + 1


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


scheduler = None

def main(config, settings):
    global scheduler

    # ─  DDP ────────────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    torch.manual_seed(123)

    # ─ Tokenizer (GPT-2 BPE, same vocab as GPT-2 pretraining) ────────────
    if is_main_process():
        print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default

    # ─ Load + clean corpus ────────────────────────────────────────────────
    if is_main_process():
        print("Loading corpus...")
    raw_text   = load_text_data()
    clean_text = clean_gutenberg(raw_text)
    split_idx  = len(clean_text) - 2048        # small held-out val set

    if is_main_process():
        print(f"Clean corpus size: {len(clean_text):,} characters")

    #  Datasets ───────────────────────────────────────────────────────────
    train_dataset = GPT2TextDataset(
        clean_text[:split_idx], tokenizer,
        max_length = config["context_length"],
        stride     = 128,
    )
    val_dataset = GPT2TextDataset(
        clean_text[split_idx:], tokenizer,
        max_length = config["context_length"],
        stride     = 128,
    )

    if is_main_process():
        print(f"Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")

    # ─ DistributedSamplers + DataLoaders ─────────────────────────────────
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = settings["batch_size"],
        sampler    = train_sampler,
        drop_last  = True,
        num_workers= 4,
        pin_memory = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = settings["batch_size"],
        sampler    = val_sampler,
        drop_last  = False,
        num_workers= 4,
        pin_memory = True,
    )


    if is_main_process():
        print(f"Loading pretrained GPT-2 ({config['variant']})...")
    model = GPT2LMHeadModel.from_pretrained(config["variant"])
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main_process():
        print(f"Model loaded  ({n_params:.1f}M parameters)\n")

 
    total_steps = len(train_loader) * settings["num_epochs"]
    warmup_steps = int(0.05 * total_steps)     # 5% warmup

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = settings["learning_rate"],
        weight_decay = settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )

    # ─Fine-tune ──────────────────────────────────────────────────────────
    train_losses, val_losses, tokens_seen, epochs_ran = finetune_model_ddp(
        model, train_loader, val_loader, optimizer, device,
        num_epochs      = settings["num_epochs"],
        eval_freq       = 5,
        eval_iter       = 1,
        start_context   = "Every effort moves you",
        tokenizer       = tokenizer,
        patience        = settings["patience"],
        min_delta       = settings["min_delta"],
        checkpoint_path = "best_finetune.pth",
    )

    # ─Post-training (rank-0 only) ────────────────────────────────────────
    if is_main_process():
        epochs_tensor = torch.linspace(0, epochs_ran, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("finetune_loss.pdf")
        print("Loss plot saved to finetune_loss.pdf")

        torch.save(model.module.state_dict(), "finetune_final.pth")
        print("Final model saved to finetune_final.pth")

    cleanup_ddp()
    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # GPT-2 small = 117M params, context 1024
    # Other options: "gpt2-medium" (345M), "gpt2-large" (774M), "gpt2-xl" (1.5B)
    FINETUNE_CONFIG = {
        "variant":        "gpt2",       # HuggingFace model name
        "context_length": 256,          # keep same as your scratch model
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-5,          # 10× smaller than scratch training
        "num_epochs":    20,            # fine-tuning converges much faster
        "batch_size":    4,             # can afford larger batch vs scratch
        "weight_decay":  0.01,
        "patience":      5,
        "min_delta":     1e-4,
    }

    main(FINETUNE_CONFIG, OTHER_SETTINGS)