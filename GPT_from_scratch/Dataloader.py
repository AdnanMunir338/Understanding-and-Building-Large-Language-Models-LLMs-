############################################ LLM _COurse_ Adnan_Munir _Dataloader ####################################

""" Run with simple "python Dataloader.py" """

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re

############################## Dataset Cleaning #########################
def clean_text(text: str) -> str:

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)        # max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)             # collapse multiple spaces
    text = re.sub(r'\t', ' ', text)                # tabs → space

    # Remove non-printable / control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Fix unicode artifacts
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # Strip leading/trailing whitespace per line
    text = '\n'.join(line.strip() for line in text.splitlines())

    # Remove very short/empty lines (noise)
    lines = [l for l in text.splitlines() if len(l) > 10]
    text = '\n'.join(lines)

    return text.strip()
 
# ─────────────────────────────────────────────
#  Shared Dataset (works with any tokenizer)
# ─────────────────────────────────────────────
 
class GPTDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []
 
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
 
    def __len__(self):
        return len(self.input_ids)
 
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
 
 
# ─────────────────────────────────────────────
#  Version 1 — BPE via tiktoken  (GPT-2 style)
# ─────────────────────────────────────────────
 
class BPETokenizerWrapper:

    def __init__(self, encoding_name: str = "gpt2"):
        self._tok = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._tok.n_vocab
 
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, allowed_special={"<|endoftext|>"})
 
    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
 
 
def bpe_dataloader(
    txt,
    batch_size  = 4,
    max_length  = 256,
    stride      = 128,
    shuffle     = True,
    drop_last   = True,
    num_workers = 0,
):

    tokenizer = BPETokenizerWrapper(encoding_name="gpt2")
    # token_ids = tokenizer.encode(txt)
    token_ids = tokenizer.encode(clean_text(txt)) ### clean text
 
    dataset    = GPTDataset(token_ids, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        drop_last   = drop_last,
        num_workers = num_workers,
    )
    return dataloader, tokenizer.vocab_size, tokenizer
 
 
# ─────────────────────────────────────────────
#  Version 2 — WordPiece via HuggingFace BERT
# ─────────────────────────────────────────────
 
class WordPieceTokenizerWrapper:

    def __init__(self, model_name: str = "bert-base-uncased"):
        self._tok = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = self._tok.vocab_size
 
    def encode(self, text: str) -> list[int]:
        # add_special_tokens=False  →  no [CLS]/[SEP] padding between chunks
        return self._tok.encode(text, add_special_tokens=False)
 
    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)
 
 
def wordpiece_dataloader(
    txt,
    batch_size  = 4,
    max_length  = 256,
    stride      = 128,
    shuffle     = True,
    drop_last   = True,
    num_workers = 0,
):

    tokenizer = WordPieceTokenizerWrapper(model_name="bert-base-uncased")
    
    # token_ids = tokenizer.encode(txt)
    token_ids = tokenizer.encode(clean_text(txt)) ### use text after cleaning
    
 
    dataset    = GPTDataset(token_ids, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        drop_last   = drop_last,
        num_workers = num_workers,
    )
    return dataloader, tokenizer.vocab_size, tokenizer



################################################################################################
if __name__ == "__main__":
    sample_text = (
        "The quick brown fox jumps over the lazy dog. " * 200
    )
 
    print("=" * 50)
    print("BPE Dataloader  (tiktoken / GPT-2)")
    print("=" * 50)
    bpe_loader, bpe_vocab, _ = bpe_dataloader(
        sample_text, batch_size=4, max_length=32, stride=16
    )
    inputs, targets = next(iter(bpe_loader))
    print(f"  Vocab size   : {bpe_vocab:,}")
    print(f"  Batches      : {len(bpe_loader)}")
    print(f"  Input shape  : {inputs.shape}")
    print(f"  Target shape : {targets.shape}")
    print(f"  Sample IDs   : {inputs[0, :8].tolist()}")
 
    print()
    print("=" * 50)
    print("WordPiece Dataloader  (BERT / HuggingFace)")
    print("=" * 50)
    wp_loader, wp_vocab, _ = wordpiece_dataloader(
        sample_text, batch_size=4, max_length=32, stride=16
    )
    inputs, targets = next(iter(wp_loader))
    print(f"  Vocab size   : {wp_vocab:,}")
    print(f"  Batches      : {len(wp_loader)}")
    print(f"  Input shape  : {inputs.shape}")
    print(f"  Target shape : {targets.shape}")
    print(f"  Sample IDs   : {inputs[0, :8].tolist()}")