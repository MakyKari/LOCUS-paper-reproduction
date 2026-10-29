# Paper data extracted from the arXiv source (2601.21082v1)

Values are read from the LaTeX tables and from the vector figures (not by eye). The rendered figures in figs/ are the paper's, unchanged (CC BY-NC-ND 4.0).

- Tables 5-6: 36 rows (6 methods).
- table7_8.csv: 24 rows; rows printed in the code's dataset order (re-ordered): minilm@512 routing, minilm@1024 routing, distilroberta@256 routing, distilroberta@512 routing, distilroberta@1024 routing, minilm@512 correctness, minilm@1024 correctness, distilroberta@256 correctness, distilroberta@512 correctness, distilroberta@1024 correctness.
- table9_10.csv: 36 rows; rows printed in the code's dataset order (re-ordered): none.

- train_sample_eff.pdf: 1 axes, worst tick-fit residual 5.9e-08 of the axis range.
  - fig1.csv: 18 points, series ['EmbedLLM', 'IRT-Net', 'LOCUS']
- perf_vs_size_and_overlap.pdf: 4 axes, worst tick-fit residual 1.0e-07 of the axis range.
  - fig4.csv: 48 points, series ['overlap_correctness', 'overlap_routing', 'size_correctness', 'size_routing']
- kth_neighbor_correctness_agreement_valtest_pooled.pdf: 1 axes, worst tick-fit residual 4.0e-08 of the axis range.
  - fig8.csv: 99 points, series ['agreement']
- model_portfolio_joint.pdf: 2 axes, worst tick-fit residual 2.0e-08 of the axis range.
  - fig9.csv: 68 points, series ['budget_coverage-greedy', 'budget_random', 'count_k-center', 'count_k-medoids', 'count_random']
- hypothetical_embeddings_recall.pdf: 1 axes, worst tick-fit residual 2.9e-08 of the axis range.
  - fig11.csv: 30 points, series ['recall@1', 'recall@10', 'recall@2', 'recall@3', 'recall@5']
- failed_model_proxy.pdf: 1 axes, worst tick-fit residual 2.4e-08 of the axis range.
  - fig12.csv: 50 points, series ['neighbour', 'router']
- convergence_overlap_fraction.pdf: 2 axes, worst tick-fit residual 8.0e-08 of the axis range.
- convergence_num_queries.pdf: 2 axes, worst tick-fit residual 1.0e-07 of the axis range.
  - fig13_14.csv: 56 points, series ['fig13_cosine', 'fig13_cosine_rank1', 'fig13_cosine_rank10', 'fig13_cosine_rank2', 'fig13_cosine_rank3', 'fig13_cosine_rank5', 'fig13_euclidean', 'fig13_euclidean_rank1', 'fig13_euclidean_rank10', 'fig13_euclidean_rank2', 'fig13_euclidean_rank3', 'fig13_euclidean_rank5', 'fig14_cosine', 'fig14_cosine_rank1', 'fig14_cosine_rank10', 'fig14_cosine_rank2', 'fig14_cosine_rank3', 'fig14_cosine_rank5', 'fig14_euclidean', 'fig14_euclidean_rank1', 'fig14_euclidean_rank10', 'fig14_euclidean_rank2', 'fig14_euclidean_rank3', 'fig14_euclidean_rank5']
- encoder_timing_multi_total.pdf: 1 axes, worst tick-fit residual 2.0e-08 of the axis range.
- decoder_timing_multi_total.pdf: 1 axes, worst tick-fit residual 4.5e-08 of the axis range.
  - fig18.csv: 270 points, series ['decoder_M1', 'decoder_M112', 'decoder_M16', 'decoder_M2', 'decoder_M32', 'decoder_M4', 'decoder_M48', 'decoder_M64', 'decoder_M8', 'decoder_M96', 'encoder_M1', 'encoder_M112', 'encoder_M16', 'encoder_M2', 'encoder_M32', 'encoder_M4', 'encoder_M48', 'encoder_M64', 'encoder_M8', 'encoder_M96']
- fig16_17.csv: 40 cells (expected 40).
- fig7.csv: 42 cells, 7 models, 7 value columns.
- fig3.csv: 42 cells; max colour mismatch 4.1/255, colour-bar resolution 0.00015.
