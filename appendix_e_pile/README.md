# Appendix E: Pile Stability and Noise Diagnostics

This is the final Appendix-E experiment requested by the user. It uses Pile
only. The previously started CIFAR diagnostics remain a draft in
`appendix_e_stability_noise/` and are not used in this appendix.

## Frozen Source Protocol

Reuse `efficiency_scalability_ablations/second_order_online`: GPT-2, 1000 fixed
Pile corpora, 128 tokens, final LayerNorm only, three training seeds, one pass
of 125 AdamW updates. Compare In-Run second order, Adam-aware second order,
Adam-aware first order and the genuine no-HVP Graph path. This is not the
older Graph-v2 second-order-cache correction or a full 100k experiment.

Graph uses semantic k=10 edges, alpha=.2, historical raw scores and in-batch
pair messages with scale matching. It has no gradient gate. Existing source
scores, training trajectories and data are never changed or retrained here.

The clean runs share identical inputs and differ by training seed. The noisy
runs also change which 10% of corpora are corrupted and their mixed/shuffled
content. Thus noisy cross-seed correlations combine data perturbation and
training variation; they are NOT pure training-seed sensitivity.

## Diagnostics

- Separate clean/noise 12x12 Spearman matrices: four methods, three seeds.
  The common original corpus IDs are aligned before ranking comparisons.
- Raw and within-method/seed standardized scores; per-seed violins compare
  corrupted versus uncorrupted members of the same noisy trajectory.
- Top-k contamination at 1/2/5/10/20/40%, including a separate graph-diverse
  selector (lambda=.05) and an explicitly oracle noise-source-blocking control.
- Directed edge orientation is neighbor source j -> score recipient i.
  Record semantic noise-source mass and per-step historical/pair messages.
- Split noise/clean message numerators at the factual scale denominator.
  These centered algebraic allocations reconstruct actual corrections but
  are not independent causal effects. Batch centering couples recipients.
- The oracle counterfactual blocks known-corrupted nodes' outgoing history
  and pair messages, recomputes scale matching, and holds raw scores, graph,
  text and training fixed. It is not denoising/retraining or a deployable method.
- Neighborhood bins cover original source-domain agreement, initial-gradient
  signed cosine agreement, positive symmetric degree and mean raw neighbor
  cosine density. Equal quantile boundaries are merged. Single-class bin
  AUROCs remain undefined rather than fabricated.
- Fixed seed-0 extrema provide corrupt demotion/promotion and uncorrupted
  promotion/demotion cases. Save actual <=128-token inputs, clean counterparts,
  five highest-weight neighbors, source IDs and the closest of the 16 actual
  utility queries by initial embedding cosine. These are selected extremes,
  not representative frequencies or proof of query causality.

Pile has no class-label analogue of CIFAR mislabel detection. Domain agreement
uses original source metadata, NOT inferred true classes for mixed corrupted
text. The corruption flags are used only for post-hoc diagnostics and the
clearly labeled oracle control, never to modify the original algorithm.

## Additional Computation

For each noisy input set, compute 768-dimensional semantic features and
1536-dimensional per-example gradients at the initial pretrained GPT-2 in
eval mode. These are diagnostic gradients, not the original time-varying
training gradients, and are never passed into the scoring path. GPUs 0/1/2
are used only to compute these caches; analyses then use the fixed scores.

## Reproduction

```bash
/home/lcheng/miniconda3/envs/esm_test/bin/python appendix_e_pile/run.py
/home/lcheng/miniconda3/envs/esm_test/bin/python appendix_e_pile/report.py
```

Initial diagnostic caches are reused; delete only a changed diagnostic cache
when intentionally rerunning it. The source paths and SHA-256 hashes are in
`results/provenance.json`. `report.py` independently verifies hashes, node and
edge counts, edge orientation, bin denominators, feature dimensions and Top-k
corruption counts. The primary run also checks score reconstruction, additive
message allocation and empty/all-source blocking endpoints.

See [the experimental report](EXPERIMENT_REPORT.md), [actual Pile cases](results/cases.md),
and [the audit](results/audit.json). Raw tables, per-seed matrices, counterfactual
arrays and five PNG/PDF/SVG figure groups are under `results/`.

The underlying source experiment had near-chance noise AUROC and no selected
subset beating its pretrained reference loss. These adverse results remain
valid and are not replaced by this diagnostic analysis. Only three seeds are
available; the three seed-pair correlations are dependent, not independent
confidence intervals.
