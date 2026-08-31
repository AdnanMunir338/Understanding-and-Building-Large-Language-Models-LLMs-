import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
#  BASE CLASS  —  shared projection + head-split logic
# ─────────────────────────────────────────────────────────────────

class _MHABase(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out          = d_out
        self.num_heads      = num_heads
        self.head_dim       = d_out // num_heads
        self.context_length = context_length

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

    def _project(self, x):
        """Project input and split into heads → (b, num_heads, T, head_dim)"""
        b, T, _ = x.shape
        Q = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        return Q, K, V

    def _merge_heads(self, x, b, T):
        """Merge heads back → (b, T, d_out)"""
        return x.transpose(1, 2).contiguous().view(b, T, self.d_out)

    def forward(self, x):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────
#  VARIANT 1 — Causal (Masked) Multi-Head Attention

class CausalMultiHeadAttention(_MHABase):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__(d_in, d_out, context_length, dropout, num_heads, qkv_bias)

        # Upper-triangular mask: position (i,j)=1 means token i must NOT attend to j
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, T, _ = x.shape
        Q, K, V = self._project(x)

        # ── Scaled dot-product scores ──────────────────────────
        scale       = K.shape[-1] ** 0.5
        attn_scores = Q @ K.transpose(2, 3) / scale          # (b, h, T, T)

        # ── Apply causal mask ──────────────────────────────────
        mask_bool = self.mask.bool()[:T, :T]                  # trim to actual T
        attn_scores = attn_scores.masked_fill(mask_bool, float('-inf'))

        # ── Softmax + dropout ──────────────────────────────────
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ── Weighted sum + merge heads ─────────────────────────
        out = self._merge_heads(attn_weights @ V, b, T)
        return self.out_proj(out)


# ──────────────────────────────────

class SlidingWindowAttention(_MHABase):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 window_size=128, qkv_bias=False):
        super().__init__(d_in, d_out, context_length, dropout, num_heads, qkv_bias)

        self.window_size = window_size

        # Build combined mask: causal  +  outside sliding window
        # mask[i, j] = True  →  token i cannot attend to token j
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1
        )
        # window mask: token i cannot attend to j if (i - j) >= window_size
        rows = torch.arange(context_length).unsqueeze(1)   # (T, 1)
        cols = torch.arange(context_length).unsqueeze(0)   # (1, T)
        window_mask = (rows - cols) >= window_size          # (T, T)

        # Combined: block if EITHER condition is true
        self.register_buffer("mask", causal_mask | window_mask)

    def forward(self, x):
        b, T, _ = x.shape
        Q, K, V = self._project(x)

        # ── Scaled dot-product scores ──────────────────────────
        scale       = K.shape[-1] ** 0.5
        attn_scores = Q @ K.transpose(2, 3) / scale          # (b, h, T, T)

        # ── Apply sliding window + causal mask ────────────────
        mask = self.mask[:T, :T]                              # trim to actual T
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # ── Softmax + dropout ──────────────────────────────────
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ── Weighted sum + merge heads ─────────────────────────
        out = self._merge_heads(attn_weights @ V, b, T)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────
#  SANITY CHECK
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparams
    B              = 2       # batch size
    T              = 16      # sequence length
    d_in           = 64      # input dim
    d_out          = 64      # output dim
    num_heads      = 4
    context_length = 64
    dropout        = 0.0     # 0 for deterministic test
    window_size    = 4       # attend to last 4 tokens only

    x = torch.randn(B, T, d_in)

    # ── Causal MHA ────────────────────────────────────────────
    causal_attn = CausalMultiHeadAttention(
        d_in, d_out, context_length, dropout, num_heads
    )
    causal_out = causal_attn(x)

    print("=" * 55)
    print("Causal Multi-Head Attention")
    print("=" * 55)
    print(f"  Input  shape : {x.shape}")
    print(f"  Output shape : {causal_out.shape}")

    # Verify causality: output at position i must not depend on i+1
    x_mod          = x.clone()
    x_mod[:, 5:, :] = torch.randn_like(x_mod[:, 5:, :])   # perturb tokens 5+
    causal_out_mod  = causal_attn(x_mod)
    causal_ok       = torch.allclose(causal_out[:, :5, :], causal_out_mod[:, :5, :], atol=1e-5)
    print(f"  Causality check (positions 0-4 unchanged) : {'✅ PASSED' if causal_ok else '❌ FAILED'}")

    print()

    # ── Sliding Window Attention ──────────────────────────────
    swa = SlidingWindowAttention(
        d_in, d_out, context_length, dropout, num_heads, window_size=window_size
    )
    swa_out = swa(x)

    print("=" * 55)
    print(f"Sliding Window Attention  (window={window_size})")
    print("=" * 55)
    print(f"  Input  shape : {x.shape}")
    print(f"  Output shape : {swa_out.shape}")

    # Verify mask pattern for one head
    with torch.no_grad():
        Q, K, _ = swa._project(x)
        scores   = (Q @ K.transpose(2, 3) / K.shape[-1] ** 0.5)
        mask     = swa.mask[:T, :T]
        scores   = scores.masked_fill(mask, float('-inf'))
        weights  = torch.softmax(scores, dim=-1)          # (B, h, T, T)

    print(f"  Mask shape   : {mask.shape}")
    print(f"  Blocked positions (row 8) : {mask[8].nonzero(as_tuple=True)[0].tolist()}")
    print(f"  Attended positions (row 8): {(~mask[8]).nonzero(as_tuple=True)[0].tolist()}")
    print(f"  → token 8 attends to tokens {(~mask[8]).nonzero(as_tuple=True)[0].tolist()} (window={window_size})")

    print()

    # ── Side-by-side param count ──────────────────────────────
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("=" * 55)
    print("Parameter count (identical — only mask differs)")
    print("=" * 55)
    print(f"  CausalMHA    : {count_params(causal_attn):,}")
    print(f"  SlidingWindow: {count_params(swa):,}")