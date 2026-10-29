# Notes for the final report (Stage 13)

Caveats, deviations and decisions to carry into REPORT.md. Numbers are in the stage summaries in results/.

## Decisions

- **The authors' code is not modified** beyond the two patches in commit f0dee58 (seeding, `--grad-accum-steps`).
  Where their protocol produces a questionable result, we keep the result and explain it here instead of changing the code.

## Stage 7 (onboarding, Fig. 3): checkpoint selection by val hit@1

- `training_script.py` saves the epoch with the best **val routing hit@1** (first max, strict `>`); this is hard-coded.
- In pools that contain Llama-3-70B-Instruct (16, 32, 64, 96 of our nested pools), the trivial "always route to the
  strongest model" already gives val hit@1 ≈ 0.599 at epoch 0, and training barely exceeds it. The saved checkpoint
  can therefore be an early, under-trained epoch:
  - pool 32: saved epoch 24, val acc 0.663 (0.682 at epoch 124);
  - pool 64: saved epoch 18, val acc 0.641 (0.688 at epoch 118);
  - pool 96: saved epoch 456 (not affected); pools 8 / 16: epochs 112 / 302.
- Consequence: rows 8-64 of Fig. 3 are 1-4 pp below the full pool (pool 64 is the worst), while the paper shows 32/64
  at or above the full pool. The "< 1% gap" claim holds for the 96-model pool only.
- Decision (user, 2026-10-27): keep the authors' protocol, no re-training with another selection rule; report this.
- The same rule affects the full-pool runs much less (best epochs 284-689 in Stage 3), but it is the reason the
  "best test at any epoch" protocol in results/stage3_summary.md is higher than the reported numbers.

## Other caveats collected so far

- **Stage 4, EmbedLLM:** ours is ~3 pp above the paper's EmbedLLM at 256-1024 anchors, even under the official
  protocol (50 epochs, lr 1e-4, last epoch: 62.77 at 1024 vs. paper 59.60). The paper's EmbedLLM (58.8-59.6) is below
  the trivial "always Llama-3-70B-Instruct" router (60.50 test hit@1). With our EmbedLLM, LOCUS vs. EmbedLLM routing
  is a tie (LOCUS wins 5/18 paired runs); LOCUS is better on correctness (16/18). Fig. 1 "4.8x" claim not reproduced.
- **Stage 8:** paper fixes n = n_ref = 4096 for the overlap experiment (PLAN.md said 1024). Fig. 13-14 model
  CodeLlama-13b-hf is not among the 112 models; CodeLlama-13b-Instruct-hf used. Identification (own z_ref nearest)
  is only 34% at overlap 0: embeddings locate the neighbourhood, not the exact model (fingerprinting claim weaker).
- **Stage 9:** Euclidean correlations below the paper (0.79 vs 0.887 Pearson) for the reference encoder; 0.81-0.85
  for 1024/2048 encoders. Fig. 7 linkage inferred as complete (merge heights equal max pairwise distances).
  Raw label Hamming distance on the same 4096 anchors correlates 0.99 with test disagreement (embedding: 0.84).
- **Stage 10:** model parameter counts are nominal sizes from model names (total 1947.5B vs paper 1930B; MoE models
  counted with all experts). Fig. 11 not reproduced at large n (synthetic per-task profile plateaus at recall@10 ≈ 0.83,
  per-dataset ≈ 0.6; paper 0.97); paper caption says "dataset-wise", text says "task-wise". Fig. 9: picking the k most
  accurate models beats k-center / k-medoids for k < 7 and coverage-greedy for most budgets.
- **Hardware:** RTX 3050 4 GB; gradient accumulation K (exact for <= 1024 anchors, equal in expectation above);
  reference encoder (4096 anchors) trained with K = 32.

## Stage 11 (EmbedLLM instability, Figs. 15-17), finished 2026-10-28

- Data: the 4096 reference anchors x 112 models; early stopping on val CE (patience 5), lr 1e-3; one seed.
- Fig. 17 (flips on test after retraining e with G frozen, same data): 2.8-6.0%, inside the paper's 1.2-8.1%.
- Fig. 16 (cosine original vs regenerated): 0.51-0.60 with 0 hidden layers (paper 0.45-0.65), but 0.68-0.88 with
  1-4 hidden layers, while the paper reports lower values for deeper MLPs (e.g. 0.270 at 3 layers, D=64). Our deeper
  models early-stop at epochs 1-9 on val CE, which may explain the opposite trend.
- Cosine distances are not comparable across methods: in the LOCUS space distinct models are themselves very close
  (CodeLlama-13b-Instruct's neighbours at cosine 0.0002-0.009), so "LOCUS 0.005 vs EmbedLLM 0.8" says little.
  Scale-free comparison on new data (overlap 0, n = 4096): flips LOCUS 1.66% vs EmbedLLM 4.57%; own model identified
  30% vs 13%. Subsample 1024: flips 1.94% vs 4.43%, identified 40% vs 33%. LOCUS is more stable, but less dramatically
  than Figs. 15-16 suggest; "distance 0 for LOCUS on the same set" is true by construction.
- Same-data regeneration of the official EmbedLLM (0 hidden, D 232) still identifies 82% of models (90% at D=512).

## Stage 5 (ablations, Tables 7-10), finished 2026-10-28

- 24 runs, seed 42, 8.7 GPU hours; mpnet / d=128 cells reuse Stage 3 seed-42 runs.
- Correctness matches the paper within 0.7 pp in 23 of 24 new cells (distilroberta within 0.05); exception albert@256
  63.57 vs paper 59.42 (paper value 7-9 pp below neighbouring cells, looks anomalous).
- Routing within 2 pp in 22 of 24 (d=256@1024 -2.36, d=512@512 -3.17).
- Spread between variants at one budget (1.4-3.3 pp; paper 1.2-3.7) is comparable to the seed spread of the default
  configuration (1.5-3.2 pp); best routing variant agrees with the paper in 1 of 6 cases. Single-seed ablation
  conclusions on routing are not supported; albert is worst on correctness everywhere (40% truncation).

## Stage 4, IRT-Net, finished 2026-10-28

- Official architecture/hyperparameters, selection by val correctness accuracy (official code), no tuning, 18 runs.
- Matches the paper: routing 60.03 / 62.70 / 63.37 / 63.90 at 256 / 512 / 1024 / 2048 (paper 59.57 / 60.17 / 63.37 /
  ~64.1); correctness 67.87 / 68.78 / 70.10 (paper 67.38 / 69.07 / 70.12). Only routing@512 is off (+2.5 pp).
- vs LOCUS (paired, 18): routing tie (LOCUS wins 10, +0.37); correctness IRT-Net better (LOCUS wins 2, -0.42),
  consistent with the paper's own Table 6 at 512/1024.
- Fig. 1 "2.3x" not reproduced: LOCUS reaches IRT-Net@2048 (63.90) only at 2048 anchors.
- Supports the EmbedLLM caveat: the other baseline reproduces to the hundredth on the same data and metrics, so the
  EmbedLLM gap is not explained by data / environment differences.

## Stage 12 (timing, Fig. 18), finished 2026-10-28

- Reference encoder, inputs on GPU, median of 50 after 10 warm-ups, fp32, RTX 3050 Laptop (3.2 GiB free).
- Encoder linear in queries (x15.4 from 256 to 4096 at 112 models) and in models; flat (3.6-5.1 ms) for 1 model.
  112 x 4096: 423 ms (paper ~104 ms on V100). Decoder 112 x 4096: 18.5 ms GPU only, 21.2 ms through the authors'
  score_queries_against_models (paper ~18 ms).
- Windows/WDDM spills to shared system memory instead of raising OOM (first attempt: 112 x 4096 took 27.8 s);
  timing.py checks the peak allocation against the memory free at start and chunks over models.

## Paper source comparison (2026-10-29)

- Paper numbers now come from the arXiv source (repro/extract_paper.py): LaTeX tables, vector-PDF figures (tick-fit
  residual < 1e-6), heatmap text (Figs. 7, 16, 17), Fig. 3 via cell colours. Comparison figures: repro/report_figures.py.
- Paper kNN@256 row in Table 5 equals the "always Llama-3-70B-Instruct" router in all 11 values (degenerate baseline).
- Paper Tables 7-8: rows minilm@512/1024 and distilroberta@all list per-dataset values in the code's dataset order
  (MMLU, GPQA, MedQA, MathQA, ASDiv, SoQA, PIQA, GSM8k, TruthQA, LogiQA), not the header order.
- Paper albert@256 correctness on PIQA is 19.50% (ours 63.5%): explains the anomalous overall 59.42.
- Fig. 3 exact: paper pools 32-96 are at or above the full pool (+0.2-0.3 pp at n=1024); full pool matches ours.
- Fig. 7: all 21 paper pairwise distances correlate 0.976 with ours (scale x1.4).
- Fig. 13: at overlap 0 our regenerated embedding moves less than the paper's (~3 vs ~12 rank-1 distances, cosine).
- Paper figures are CC BY-NC-ND 4.0: shown unchanged in <details> blocks, never drawn over.
