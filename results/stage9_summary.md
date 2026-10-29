# Stage 9: embedding geometry

Reference encoder = Stage 6 bank (runs/ref/a4096_s42). `raw_labels_a4096` is a no-encoder control: the normalized Hamming distance between the models' labels on the same 4096 anchors.

## Table 3: distance vs. test disagreement over 6216 model pairs

| embedding | distance | pearson | spearman | kendall | paper_pearson | paper_spearman | paper_kendall |
|---|---|---|---|---|---|---|---|
| ref_a4096 | cosine | 0.838 | 0.845 | 0.654 | 0.845 | 0.886 | 0.714 |
| ref_a4096 | euclidean | 0.792 | 0.794 | 0.593 | 0.887 | 0.876 | 0.702 |
| a1024_s42 | cosine | 0.833 | 0.839 | 0.652 |  |  |  |
| a1024_s42 | euclidean | 0.813 | 0.814 | 0.622 |  |  |  |
| a1024_s43 | cosine | 0.824 | 0.850 | 0.669 |  |  |  |
| a1024_s43 | euclidean | 0.846 | 0.838 | 0.650 |  |  |  |
| a1024_s44 | cosine | 0.871 | 0.858 | 0.676 |  |  |  |
| a1024_s44 | euclidean | 0.810 | 0.800 | 0.605 |  |  |  |
| a2048_s42 | cosine | 0.860 | 0.854 | 0.674 |  |  |  |
| a2048_s42 | euclidean | 0.847 | 0.836 | 0.650 |  |  |  |
| a2048_s43 | cosine | 0.858 | 0.840 | 0.656 |  |  |  |
| a2048_s43 | euclidean | 0.825 | 0.820 | 0.627 |  |  |  |
| a2048_s44 | cosine | 0.867 | 0.854 | 0.671 |  |  |  |
| a2048_s44 | euclidean | 0.820 | 0.820 | 0.626 |  |  |  |
| raw_labels_a4096 | hamming | 0.990 | 0.986 | 0.902 |  |  |  |

Distance ranges (reference encoder): cosine 0.000–1.002, Euclidean 0.07–35.51 (paper Fig. 6: cosine 0–0.9, Euclidean 0–22).

## Fig. 7: Euclidean distances among math / code models

| model_a | model_b | same_family | euclidean | paper |
|---|---|---|---|---|
| MetaMath-Mistral-7B | llemma_34b | True | 10.52 | 6.59 |
| MetaMath-Mistral-7B | deepseek-math-7b-instruct | True | 14.01 | 7.22 |
| MetaMath-Mistral-7B | deepseek-coder-1.3b-base | False | 28.73 | 18.47 |
| MetaMath-Mistral-7B | octocoder | False | 26.57 |  |
| MetaMath-Mistral-7B | deepseek-coder-6.7b-instruct | False | 20.88 |  |
| MetaMath-Mistral-7B | CodeLlama-7b-hf | False | 22.77 |  |
| llemma_34b | deepseek-math-7b-instruct | True | 6.01 | 1.87 |
| llemma_34b | deepseek-coder-1.3b-base | False | 22.26 |  |
| llemma_34b | octocoder | False | 19.32 |  |
| llemma_34b | deepseek-coder-6.7b-instruct | False | 11.35 |  |
| llemma_34b | CodeLlama-7b-hf | False | 14.12 |  |
| deepseek-math-7b-instruct | deepseek-coder-1.3b-base | False | 19.57 |  |
| deepseek-math-7b-instruct | octocoder | False | 16.76 |  |
| deepseek-math-7b-instruct | deepseek-coder-6.7b-instruct | False | 9.68 |  |
| deepseek-math-7b-instruct | CodeLlama-7b-hf | False | 11.90 |  |
| deepseek-coder-1.3b-base | octocoder | True | 4.44 | 1.66 |
| deepseek-coder-1.3b-base | deepseek-coder-6.7b-instruct | True | 13.97 |  |
| deepseek-coder-1.3b-base | CodeLlama-7b-hf | True | 11.02 |  |
| octocoder | deepseek-coder-6.7b-instruct | True | 10.23 |  |
| octocoder | CodeLlama-7b-hf | True | 6.89 |  |
| deepseek-coder-6.7b-instruct | CodeLlama-7b-hf | True | 3.97 | 2.97 |

- complete linkage: merge heights 3.97, 4.44, 6.01, 13.97, 14.01, 28.73; families separate at 2 clusters: True
- average linkage: merge heights 3.97, 4.44, 6.01, 10.53, 12.27, 18.66; families separate at 2 clusters: True
- ward linkage: merge heights 3.97, 4.44, 6.01, 13.88, 14.72, 32.72; families separate at 2 clusters: True
- paper (complete linkage, from the dendrogram): 1.66, 1.87, 2.97, 7.22, 8.17, 18.47

## Fig. 8: agreement with the k-th nearest neighbour (cosine), val+test pooled, mean over models

| embedding | k=1 | k=5 | k=20 | k=50 | k=99 | k=111 |
|---|---|---|---|---|---|---|
| paper (plot ≈) | 0.790 |  | 0.750 | 0.720 | 0.620 |  |
| a1024_s42 | 0.789 | 0.765 | 0.745 | 0.705 | 0.635 | 0.596 |
| a1024_s43 | 0.787 | 0.776 | 0.740 | 0.714 | 0.625 | 0.566 |
| a1024_s44 | 0.798 | 0.777 | 0.738 | 0.711 | 0.629 | 0.563 |
| a2048_s42 | 0.789 | 0.774 | 0.739 | 0.706 | 0.630 | 0.563 |
| a2048_s43 | 0.790 | 0.771 | 0.736 | 0.713 | 0.633 | 0.562 |
| a2048_s44 | 0.794 | 0.779 | 0.741 | 0.712 | 0.635 | 0.595 |
| raw_labels_a4096 | 0.834 | 0.789 | 0.742 | 0.709 | 0.624 | 0.552 |
| ref_a4096 | 0.802 | 0.776 | 0.741 | 0.714 | 0.636 | 0.564 |
