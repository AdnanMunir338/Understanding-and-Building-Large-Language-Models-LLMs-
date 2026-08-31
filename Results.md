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

## EXP2
<img width="778" height="220" alt="image" src="https://github.com/user-attachments/assets/7ecf3484-c51a-462f-966f-ecfb8f4e7ec6" />

## Exp3
<img width="776" height="218" alt="image" src="https://github.com/user-attachments/assets/36a8a548-19c9-4e8a-986c-0f0a25513cb5" />

## Exp4
<img width="767" height="229" alt="image" src="https://github.com/user-attachments/assets/30e25cdf-d564-4c6b-97b3-11a66351455c" />

## Exp5
<img width="760" height="217" alt="image" src="https://github.com/user-attachments/assets/687b49a6-7dac-43b9-bf68-9470adcb14b9" />

## Exp6
<img width="707" height="226" alt="image" src="https://github.com/user-attachments/assets/94bbae39-8512-4ab0-99cd-efd2f0a05aab" />

## Exp7
<img width="706" height="218" alt="image" src="https://github.com/user-attachments/assets/3a93016c-359e-4156-aa93-e2f89ccdab4f" />

## Exp8
<img width="691" height="180" alt="image" src="https://github.com/user-attachments/assets/8f92e856-effd-4135-8d89-48c1a4e21458" />

## Exp9
<img width="727" height="244" alt="image" src="https://github.com/user-attachments/assets/0542e64b-497b-407f-a1fd-deabd4ba6593" />

## Exp10 
<img width="694" height="206" alt="image" src="https://github.com/user-attachments/assets/2a21331d-5c47-4773-945a-0c03cb75b0ad" />

