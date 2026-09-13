# Pile 100k-corpus main-table experiment

This experiment uses 100,000 training corpora and 10,000 disjoint validation
corpora sampled from `monology/pile-uncopyrighted`. The exact token count is
recorded in `data/Pile_large/manifest.json`; it is not the paper's 10B-token setting.

Each corpus is truncated to 128 GPT-2 tokens for attribution and retraining.
Attribution follows one real AdamW pass while differentiating GPT-2's final
LayerNorm (1,536 trainable attribution parameters). Selection evaluation fine-tunes all GPT-2 parameters for one epoch and reports
validation next-token accuracy as the primary metric, while retaining validation
loss as a diagnostic. Noise AUROC uses 10% token-mixture
corruption. Domain nRecall@1K is macro averaged over domains with at least 20
training and 5 validation corpora.

The four In-Run scores are trajectory scores. Graph v2 uses a 10-NN GPT-2
embedding graph, gradient-agreement confidence gating, historical neighborhood
diffusion, and an in-batch pair surrogate for the second-order variant. Selection
maximizes value minus symmetric-graph redundancy with fixed alpha 0.2 and
scale-relative lambda 0.05; these values predate the final 100k evaluation.

Graph v2 raw selection cells are stored under the raw_100k_graph_v2 directories;
frozen static cells remain under raw_100k. Selection accuracy is stored as a 0-1
proportion in raw JSON and displayed as a percentage in the main table so small
but nonzero standard deviations are visible. Standard deviations are population
values (ddof=0) over seeds 0, 1, and 2. Exact zero is expected for deterministic,
seed-independent retrieval methods.

## Graph v5 ablation (not the main result)

This failed ablation tests full semantic-neighborhood diffusion over the frozen
10-NN graph. Two scale-aligned steps use the pre-fixed alpha 0.2; subset selection
uses symmetric sparse-graph greedy selection with lambda fraction 0.2. Non-Graph
rows remain frozen. It improves stability/noise AUROC but degrades Domain Recall, so the main table remains on coherent Graph v2. Raw v5 cells are under `raw_100k_graph_v5`, and superseded
v2-v4 artifacts are retained for auditability.
