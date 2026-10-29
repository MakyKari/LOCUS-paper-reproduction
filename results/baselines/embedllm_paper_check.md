# Is the paper's EmbedLLM a 'single best model' router?

Test routing hit@1, %, overall and per dataset. Rows 'best on the anchors' always route to the model with the highest accuracy on the run's anchors (mean over seeds 42-44).

| router | anchors | overall | MathQA | LogiQA | MedQA | PIQA | TruthQA | MMLU | GSM8k | GPQA | ASDiv | SoQA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always meta-llama__Meta-Llama-3-70B-Instruct | - | 60.50 | 59.92 | 47.06 | 75.42 | 85.07 | 41.89 | 78.27 | 89.29 | 29.00 | 10.10 | 30.86 |
| EmbedLLM (paper) | 256 | 58.80 | 58.23 | 41.18 | 72.03 | 79.85 | 35.14 | 75.55 | 88.39 | 28.20 | 19.19 | 30.25 |
| always the best model on the anchors: a256_s42=meta-llama__Meta-Llama-3-70B, a256_s43=meta-llama__Meta-Llama-3-70B-Instruct, a256_s44=meta-llama__Meta-Llama-3-70B-Instruct | 256 | 59.62 | 56.82 | 43.79 | 74.48 | 85.32 | 36.49 | 77.25 | 86.90 | 28.00 | 14.81 | 30.86 |
| EmbedLLM (ours) | 256 | 62.38 | 51.76 | 44.44 | 67.23 | 85.32 | 47.30 | 80.87 | 81.85 | 27.73 | 49.16 | 36.01 |
| EmbedLLM (paper) | 512 | 59.47 | 59.49 | 45.10 | 74.29 | 77.61 | 43.24 | 76.83 | 89.29 | 28.80 | 12.12 | 29.63 |
| always the best model on the anchors: a512_s42=meta-llama__Meta-Llama-3-70B-Instruct, a512_s43=CausalLM__34b-beta, a512_s44=CausalLM__34b-beta | 512 | 58.77 | 46.41 | 45.75 | 62.81 | 86.07 | 36.49 | 83.59 | 78.57 | 26.87 | 4.04 | 30.86 |
| EmbedLLM (ours) | 512 | 62.46 | 50.77 | 45.10 | 63.75 | 86.32 | 50.45 | 82.97 | 83.63 | 25.67 | 48.48 | 34.77 |
| EmbedLLM (paper) | 1024 | 59.60 | 58.65 | 50.98 | 72.88 | 79.10 | 47.30 | 76.83 | 89.29 | 23.80 | 25.25 | 30.86 |
| always the best model on the anchors: a1024_s42=meta-llama__Meta-Llama-3-70B-Instruct, a1024_s43=meta-llama__Meta-Llama-3-70B-Instruct, a1024_s44=meta-llama__Meta-Llama-3-70B | 1024 | 59.62 | 56.82 | 43.79 | 74.48 | 85.32 | 36.49 | 77.25 | 86.90 | 28.00 | 14.81 | 30.86 |
| EmbedLLM (ours) | 1024 | 62.96 | 51.20 | 45.10 | 67.33 | 86.57 | 50.90 | 82.40 | 82.74 | 24.53 | 57.74 | 32.10 |

## Mean |difference| to 'always the best model' over the 10 datasets, pp

| anchors | paper_vs_always_best_mean_abs | ours_vs_always_best_mean_abs |
|---|---|---|
| 256 | 3.71 | 8.01 |
| 512 | 1.72 | 8.86 |
| 1024 | 4.09 | 9.33 |
