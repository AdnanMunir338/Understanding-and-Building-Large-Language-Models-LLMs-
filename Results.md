# Experimental Results

## Loss Curve Summary

| Exp | Final Train Loss | Final Val Loss | Behavior |
|---|---|---|---|
| 1–4 | ~0.1 | ~7.5 | Overfitting on small data |
| 5 | ~1.5 (converges ~ep.12) | rises after ep.12 | Overfitting past convergence point |
| 6 | ~0 | ~7.5 | Severe overfitting (large model, small data) |
| 7 | ~5.0 | ~4.5 | Stable convergence (large model, large data) |
| 8 | ~2.0 | ~2.9 | Fine-tuning improves coherence |
| 9 | ~0.4 | ~1.4 (plateau after ep.1) | Instruction-tuning overfit signal |
| 10 | ~2.4 | ~3.3 (smoother) | Regularized instruction-tuning |

## Sample Generations

**Prompt:** "Every effort moves you closer to"

- **Exp1 (BPE, Causal MHA):** *"...to degree he had the same quality as his pictures—the quality of looking cleverer than he was..."*
- **Exp9 (Instruction-tuned):** Successfully follows factual/creative instructions but hallucinates on knowledge queries (e.g., fabricated Sherlock Holmes biography).
- **Exp10 (Regularized instruction-tuned):** Improved fluency; hallucination and stylistic bleeding reduced but not eliminated.

## EXP1
<img width="758" height="227" alt="image" src="https://github.com/user-attachments/assets/104aa84c-6096-4511-8a9f-657d6cbc9afe" />

