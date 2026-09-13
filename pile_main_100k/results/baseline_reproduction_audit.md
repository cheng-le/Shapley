# Pile 100k Baseline Reproduction Audit

## Verdict

The current Pile 100k table is a local proxy benchmark, not a strict reproduction
of every cited paper. It must not describe all rows as official implementations or
claim state-of-the-art superiority from these comparisons. Selection retraining is
real full-parameter GPT-2 training, but most attribution baselines use the shared
1,536-dimensional final-LayerNorm cache.

| Row | Current implementation | Reproduction status | Required strict replacement |
|---|---|---|---|
| Random | Seeded Gaussian ranking | Valid control | None |
| BM25 | Local Okapi BM25 over whitespace tokens | Conventional baseline, not library-identical | Freeze tokenizer/query protocol and validate against a standard implementation |
| KNN-Shapley | Training-query embedding cosine similarity | **Not KNN-Shapley** | Implement the KNN-Shapley recurrence and utility/label definition |
| Influence Function | Training-gradient/query-gradient dot product | **Not Influence Functions** | Add inverse-Hessian-vector product and damping/convergence diagnostics |
| TRAK (1 ckpt) | Gradient dot product divided by training-gradient norm | **Not TRAK** | Use TRAK projection, kernel solve, checkpoint/model ensemble, and official protocol |
| LESS | Normalized gradient cosine | Partial low-dimensional proxy | Use optimizer-aware influence features, checkpoint aggregation, and the published selection protocol |
| 1st Order In-Run | Per-batch validation-gradient alignment on one LayerNorm trajectory | Local approximation | Validate signs/scales and full paper protocol |
| 2nd Order In-Run | LayerNorm HVP correction capped at 0.25x first-order scale | Local approximation | Justify or remove the unpublished curvature cap |
| Adam-aware 1st/2nd | Sample-conditioned Adam directions on LayerNorm state | Local Adam-aware approximation | Verify equation-by-equation against the reference implementation |
| Graph v2 | Random cache replay of one final score per sample | Paper-inspired approximation | Record true step order and per-step contributions during scoring |

## Concrete Code Evidence

- `common.py:405`: KNN-Shapley is assigned plain semantic cosine.
- `common.py:406`: Influence is assigned the first-order gradient dot product.
- `common.py:407-408`: TRAK and LESS are normalized variants of the same dot product.
- `domain_recall/run.py:58-61`: first/second-order rows are overwritten with identical arrays.
- `build_graph_v2_scores.py:22-33`: Graph reconstructs a random trajectory from final cached scores.
- `trajectory.py:56-59`: second-order curvature is clipped by a local heuristic.

## Main-Table Policy

Graph v5 is retained as a failed ablation: it improves score stability and noise
AUROC but severely reduces Domain Recall. The main table is restored to Graph v2,
the closest currently executable version to the draft. No result is altered or
selected cell-by-cell. Achieving a prescribed win count is not a valid evaluation
criterion; methods and hyperparameters must be frozen before final metrics.
