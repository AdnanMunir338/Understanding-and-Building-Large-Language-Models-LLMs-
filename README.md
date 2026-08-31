# Understanding and Building Large Language Models

**Author:** Adnan Munir ([Adnan.munir@liu.se](mailto:Adnan.munir@liu.se))

This repository documents the end-to-end development of a GPT-style Large Language Model (LLM) built from scratch — covering tokenization, causal and sliding-window attention, pretraining, fine-tuning, instruction-tuning, and fairness evaluation.

Reference notebook: [Causal Self-Attention on Kaggle](https://www.kaggle.com/code/aisuko/causal-self-attention)

---

## 📌 Overview

The pipeline progresses through four stages:

1. **Pre-processing** — dataset cleaning, tokenization (BPE vs WordPiece), data loader construction
2. **Attention mechanisms** — Causal Multi-Head Attention vs Sliding-Window Multi-Head Attention
3. **Pretraining & scaling experiments** — model size vs dataset size trade-offs
4. **Fine-tuning** — domain adaptation and instruction-following
5. **Fairness evaluation** — bias analysis across race, nationality, gender, and religion

---

## 1. Pre-Processing

### 1.1 Dataset Cleaning
A deterministic preprocessing pipeline (whitespace regulation, control character stripping, length filtering) separates high-quality linguistic data from raw text noise.

### 1.2 Byte-Pair Encoding (BPE) — GPT-2 Style
Iteratively merges the most common adjacent symbol pairs starting at the byte level, eliminating out-of-vocabulary (OOV) issues entirely.

### 1.3 WordPiece Encoding
Builds vocabulary using a maximum-likelihood criterion rather than pure frequency, prioritizing subwords most predictive of sentence structure.

| Tokenizer | Vocab Size | Sample IDs (first 8) |
|---|---|---|
| BPE (tiktoken/GPT-2) | 50,257 | 7586, 21831, 18045, 625, 262, 16931, 3290, 13 |
| WordPiece (BERT/HuggingFace) | 30,522 | 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971 |

### 1.4 Causal Multi-Head Attention
Applies an upper-triangular mask (via masked fill → `-inf` → softmax) to enforce strict left-to-right autoregressive prediction.

### 1.5 Sliding-Window Multi-Head Attention
Reduces complexity from quadratic to linear by combining the causal mask with a local lookback window (logical OR of "future" and "out-of-window" masks), enabling longer sequences with lower memory overhead.

---

## 2. Experiments

| Exp | Tokenizer | Attention | Data/Model Scale | Notes |
|---|---|---|---|---|
| 1 | BPE | Causal MHA | Small model / Small data | Baseline |
| 2 | WordPiece | Causal MHA | Small model / Small data | Comparable to Exp1 |
| 3 | BPE | Sliding-Window MHA | Small model / Small data | ~Same as Exp1, masking differs |
| 4 | WordPiece | Sliding-Window MHA | Small model / Small data | ~Same as Exp2 |
| 5 | BPE | Causal MHA | Small model / **Large data** (8 books) | Converges ~epoch 12, then overfits |
| 6 | BPE | Causal MHA | **Large model** / Small data | Severe overfitting (train loss →0, val loss plateaus ~7.5) |
| 7 | BPE | Causal MHA | **Large model** / **Large data** | Stable convergence; train loss ~5.0, val loss ~4.5 |
| 8 | — | — | Fine-tune on curated 8-book corpus (~1.6M tokens) | Marked coherence/style improvement |
| 9 | — | — | Instruction fine-tune (Alpaca + literary hybrid) | Learns instruction-following; hallucinations + stylistic bleeding emerge |
| 10 | — | — | Instruction fine-tune + response-only loss masking, label smoothing, cosine LR, early stopping | Fluency improves; hallucination/bleeding reduced but not eliminated |

### Key Findings
- **Causal vs Sliding-Window attention** produce comparable outputs — the only functional difference is the masking strategy.
- **Small datasets overfit regardless of model scale** (Exp 5, 6); a larger corpus (Exp 7) enables more stable convergence.
- **Fine-tuning pretrained weights** (Exp 8) is far more sample/compute-efficient than training from scratch for domain adaptation.
- **Instruction fine-tuning** (Exp 9–10) introduces new failure modes — factual hallucination and stylistic bleeding across source books — that training-objective tweaks alone (loss masking, label smoothing, weight decay, cosine LR) only partially resolve. Root cause: GPT-2's small capacity, 256-token context window, and mixed dataset composition.

---

## 3. Fairness Evaluation

Bias was measured via sentiment (VADER compound), toxicity, and regard (positive/negative) across four demographic axes.

| Category | Key Result |
|---|---|
| **Race** | Most balanced overall (sentiment: White ~0.62, Black ~0.67, Asian ~0.53, Hispanic ~0.58); Black-group toxicity outlier (~0.013, high variance) |
| **Nationality** | Chinese/British score higher sentiment (~0.5) vs American/Indian (~0.2); reflects English-language Western pretraining bias |
| **Gender** | Male highest positive sentiment/regard (~0.5/0.57); female highest toxicity (~0.008); non-binary lowest on both sentiment and regard |
| **Religion** | **Most biased category** — largest sentiment gap (Christian ~0.8 vs Muslim ~0.15), elevated Muslim toxicity, lowest Jewish positive regard |

---

## 4. Conclusion

This project demonstrates a full LLM development lifecycle — from scratch pretraining through fine-tuning to fairness auditing. Two attention mechanisms (causal, sliding-window) were shown to be functionally equivalent apart from masking. Dataset scale was found to matter more than model scale for avoiding overfitting. Fine-tuning pretrained weights significantly accelerated domain and instruction adaptation, but exposed hallucination and stylistic bleeding that persisted despite targeted regularization — pointing to fundamental capacity/context-length limits of GPT-2-scale models. The fairness evaluation revealed inherited pretraining biases, most notably against Muslim and non-binary groups, underscoring the need for explicit dataset curation and alignment work.

---

## References
1. S. Raschka, "rasbt," GitHub. https://github.com/rasbt
2. F. Heintz, "Understanding and Building Large Language Models (IDA)," Linköping University. https://www.ida.liu.se/~frehe08/llm/
3. Y. Chang et al., "A survey on evaluation of large language models," *ACM TIST*, vol. 15, no. 3, pp. 1–45, 2024.

## Acknowledgements
Built using the Alvis HPC cluster. Generative LLMs were used as assistive tools for paraphrasing/organizing text; all experimental design, implementation, and analysis were conducted independently.
