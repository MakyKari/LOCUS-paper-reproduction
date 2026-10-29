# Stage 4: LOCUS vs. baselines on the same anchor sets

Test metrics in %, mean over 3 seeds; every method selects its checkpoint / hyperparameter by val hit@1, except IRT-Net (val correctness accuracy, as in its official code).

Trivial router (always `meta-llama__Meta-Llama-3-70B-Instruct`, the best model by train accuracy): routing 60.50.

## Mean over seeds

| metric | anchors | LOCUS | EmbedLLM | EmbedLLM+ | IRT-Net | K-Means | kNN |
|---|---|---|---|---|---|---|---|
| routing | 128 | 60.82 | 59.51 | 59.84 | 57.94 | 61.24 | 60.07 |
| routing | 256 | 61.73 | 62.38 | 61.21 | 60.03 | 62.33 | 61.17 |
| routing | 512 | 61.18 | 62.46 | 62.18 | 62.70 | 61.72 | 61.00 |
| routing | 768 | 62.58 | 63.06 | 62.46 | 62.69 | 63.56 | 61.22 |
| routing | 1024 | 62.46 | 62.96 | 63.26 | 63.37 | 63.04 | 62.20 |
| routing | 2048 | 64.07 | 64.23 | 65.24 | 63.90 | 64.38 | 63.66 |
| correctness | 128 | 65.59 | 64.79 | 64.98 | 65.88 | 61.78 | 62.15 |
| correctness | 256 | 67.25 | 66.98 | 67.19 | 67.87 | 61.96 | 63.50 |
| correctness | 512 | 68.50 | 68.29 | 67.14 | 68.78 | 67.67 | 66.28 |
| correctness | 768 | 69.21 | 68.54 | 67.99 | 69.57 | 67.25 | 66.24 |
| correctness | 1024 | 69.49 | 68.94 | 68.31 | 70.10 | 69.28 | 68.93 |
| correctness | 2048 | 70.32 | 69.66 | 69.55 | 70.66 | 69.82 | 70.12 |

## Paired by (anchors, seed): LOCUS minus baseline, 18 pairs

| vs | metric | pairs | locus_wins | mean_diff | min_diff | max_diff |
|---|---|---|---|---|---|---|
| EmbedLLM | routing | 18 | 5 | -0.29 | -3.20 | 2.33 |
| EmbedLLM | correctness | 18 | 16 | 0.53 | -0.96 | 2.25 |
| EmbedLLM+ | routing | 18 | 7 | -0.23 | -2.43 | 2.10 |
| EmbedLLM+ | correctness | 18 | 17 | 0.86 | -0.60 | 1.67 |
| IRT-Net | routing | 18 | 10 | 0.37 | -3.37 | 3.27 |
| IRT-Net | correctness | 18 | 2 | -0.42 | -1.09 | 0.08 |
| K-Means | routing | 18 | 5 | -0.57 | -3.33 | 2.07 |
| K-Means | correctness | 18 | 14 | 2.10 | -0.33 | 6.86 |
| kNN | routing | 18 | 13 | 0.59 | -2.70 | 2.80 |
| kNN | correctness | 18 | 17 | 2.19 | -0.05 | 6.98 |

## Fig. 1 claim

- EmbedLLM @ 2048 anchors (ours): 64.23  (paper Fig. 1: ≈62.6). Smallest LOCUS budget reaching it: ours none of 128-2048, paper's own numbers 512.
- IRT-Net @ 2048 anchors (ours): 63.90  (paper Fig. 1: ≈64.1). Smallest LOCUS budget reaching it: ours 2048, paper's own numbers 1024.
