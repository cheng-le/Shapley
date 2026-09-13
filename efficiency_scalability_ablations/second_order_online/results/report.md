# True HVP vs Hessian-Free Graph

1000 Pile corpora, GPT-2 final LayerNorm only, 128 tokens, 125 AdamW updates (batch=8), three seeds. All four methods follow bitwise-identical optimizer trajectories within each seed/scenario. Graph consumes first-order Adam-aware scores online and makes ZERO higher-order gradient/HVP calls. No cached second-order scores or future gradients are read.

Each HVP baseline computes ONE HVP per batch, not a dense Hessian or one HVP per example. SGD-style In-Run attribution is evaluated along the same AdamW training trajectory. The existing 25% curvature-SD cap is retained for both HVP baselines. Graph uses the paper's semantic weights without the optional gradient gate.

## Clean Timing

Mean (sample SD), seconds. Interaction includes HVP plus curvature correction, or graph diffusion + pair correction including device-to-host score transfer. It excludes the baseline's additional differentiable-query preparation, which IS included in attribution and pipeline totals.

| Method | Interaction | Attribution + graph prep | Training/attribution pipeline + graph prep | Online CUDA peak MiB |
|---|---:|---:|---:|---:|
| inrun2 | 2.671 (0.033) | 12.310 (0.128) | 14.767 (0.167) | 3316.099 (0.000) |
| adam2 | 2.690 (0.014) | 12.339 (0.098) | 14.770 (0.125) | 3316.146 (0.000) |
| graph | 0.037 (0.001) | 11.508 (0.085) | 13.939 (0.098) | 1756.634 (0.000) |
| adam1 | 0.001 (0.000) | 9.646 (0.029) | 12.078 (0.024) | 1756.009 (0.000) |

Graph prep includes a fresh full-corpus embedding forward pass and exact KNN. Attribution totals exclude the common optimizer-update pass; pipeline totals include it. Model loading, tokenization, two warmup steps and utility retraining are excluded for all arms. Graph preparation peak memory and CPU RSS are retained in raw JSON. CPU RSS is sampled every 5 ms.

## Quality

Noise AUROC: separate 10%-corrupted training run, three seeds. Subset validation loss: retrain from the SAME pretrained initialization on 100 selected clean examples, one pass, 13 updates; evaluate on 128 disjoint validation examples (not the 16 attribution utility examples). Top-k is used for ALL primary methods to isolate score quality. Graph-diverse is a separate lambda=.05 selection arm. Loss is mean per-example token-averaged NLL.

| Method | Noise AUROC | Subset validation loss | Reference coverage |
|---|---:|---:|---:|
| inrun2 | 0.49442 (0.03277) | 3.55247 (0.00054) | 0.77733 (0.02641) |
| adam2 | 0.50739 (0.00878) | 3.55407 (0.00077) | 0.77667 (0.00379) |
| graph | 0.50873 (0.01198) | 3.55310 (0.00218) | 0.75333 (0.01650) |
| adam1 | 0.50790 (0.00781) | 3.55366 (0.00252) | 0.77533 (0.00751) |
| graph_diverse | 0.50873 (0.01198) | 3.55270 (0.00176) | 0.75567 (0.02013) |

## Boundaries

- This is a real 1k online/training benchmark, not a measured full 100k trajectory. Previous 1k/10k/50k/100k cached-graph scalability results remain separate; do not extrapolate measured end-to-end speedups to 100k.
- Only 1536 LayerNorm parameters are differentiated. Full-parameter GPT-2/CIFAR conclusions need additional experiments. Higher-order autograd is real but restricted to this parameter set.
- The graph interaction is a surrogate, not a numerically equivalent HVP. Score agreement, AUROC and held-out subset loss must be reported with speed; no outcome-based tuning or filtering is performed.
- Stage synchronization adds instrumentation overhead. Three seeds use separate RTX 3090 devices, with method order rotated within seed; methods share the same device within each paired comparison.
- A large interaction-only speedup need not imply a large end-to-end speedup because common gradient extraction/training can dominate. Report both, including one-time graph prep.
- The new Hessian-free Graph is distinct from the old Graph-v2 path that consumed Adam-aware second-order caches. Old performance numbers cannot be relabeled as Hessian-free results.
