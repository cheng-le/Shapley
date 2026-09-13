# Robustness and Attribution Stability

## Purpose

This experiment tests whether graph-aware valuation is more robust to corrupted
training examples and more stable across random seeds than its pointwise
Adam-aware counterpart. It is the experiment group proposed in Section 4.3 of
`PROJECT_MIGRATION_PROMPT.md`, following the Pile main-table experiment.

The analysis answers three questions:

1. Does a graph-aware score improve corrupted-example detection AUROC and reduce
   the fraction of corrupted examples among high-value selections?
2. Does it improve cross-seed rank consistency on the fixed Pile corpus?
3. Does graph propagation suppress, rather than promote, corrupted nodes, and
   how does this behavior vary with gradient-agreement confidence?

## Models and Data

- **Pile:** the fixed 100,000-train/10,000-validation corpus from
  `monology/pile-uncopyrighted`, GPT-2 124M, sequence length 128, and attribution
  over the final LayerNorm. Ten percent of training corpora are independently
  corrupted for seeds 0, 1, and 2. Graph v2 is a 10-NN cosine graph over pooled
  GPT-2 representations with gradient-agreement gating.
- **CIFAR-10:** three independently sampled 1,000-image training subsets with
  exactly 10% incorrect labels, using ImageNet-pretrained ResNet-18 and the
  completed Table 4/Table 5 score caches. The graph uses 10 nearest neighbors in
  the ResNet-18 feature space.

The primary paired comparisons are Adam-aware first/second order versus Graph
Adam-aware first/second order. Pile robustness metrics are additionally emitted
for every method present in the authoritative noise cache.

## Metrics and Protocol

- Corruption AUROC uses negative data value as the anomaly score.
- High-value contamination is measured at Pile 1/5/10/20% and CIFAR 10/20/40%.
- Pile cross-seed stability reports pairwise Spearman, Kendall tau, Top-k overlap,
  and standardized score dispersion. Itemwise cross-seed correlation is not
  computed for CIFAR because each seed samples a different 1,000-image subset.
- Corrupted-node score changes compare each Graph score with its paired raw
  Adam-aware score. Pile changes are also stratified by graph-confidence quartile.
- Summary deviations use population standard deviation (`ddof=0`).

This runner is read-only with respect to source experiment artifacts. It uses
existing caches and does not retrain GPT-2 or ResNet-18. The Pile Graph v2 cache
remains a paper-inspired trajectory-replay approximation, as documented in
`pile_main_100k/results/baseline_reproduction_audit.md`; these diagnostics do not
turn it into a strict implementation of the draft's online algorithm.

## Run

From the repository root:

```bash
python -m robustness_attribution_stability.run_experiment
```

Outputs are written under `robustness_attribution_stability/results/`.
