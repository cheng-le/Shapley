# Efficiency, Scalability, and Ablations

This is a cached-score component benchmark, not end-to-end attribution or utility retraining.
Timing: seed-0 clean cache, three process-isolated repetitions per scale. Ablations: three training-cache seeds, a fixed nested 10k subset of the 100k corpus, 10% selection.

## Graph construction

| Corpora | Exact GPU seconds | HNSW CPU seconds | ANN Recall@10 | Graph cache MiB |
|---:|---:|---:|---:|---:|
| 1000 | 0.004 (0.001) | 0.068 (0.002) | 0.9991 | 0.126 |
| 10000 | 0.042 (0.003) | 0.976 (0.023) | 0.9979 | 1.259 |
| 50000 | 0.411 (0.000) | 8.524 (0.029) | 0.9972 | 6.294 |
| 100000 | 1.355 (0.009) | 20.640 (0.421) | 0.9963 | 12.589 |

Graph cache is uncompressed array payload (neighbors, semantic/gated weights, confidence, two scores), excluding embeddings, gradients and the ANN index.

## All mechanism diagnostics

Mean (sample SD), three cache seeds. AUROC uses negative contribution to detect corrupted examples. Pair affinity and coverage use the SAME semantic k=10 reference graph within each seed.

| Variant | Order | Noise AUROC | Selected noise | Pair affinity | Coverage | Rank stability |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 | 0.55376 (0.01010) | 0.10500 (0.00300) | 0.06127 (0.00365) | 0.74557 (0.00905) | 0.21159 |
| full | 2 | 0.55334 (0.01050) | 0.10433 (0.00153) | 0.06200 (0.00331) | 0.74587 (0.01049) | 0.21314 |
| k5 | 1 | 0.55190 (0.00900) | 0.10400 (0.00173) | 0.05993 (0.00283) | 0.75180 (0.00978) | 0.19966 |
| k5 | 2 | 0.55158 (0.00928) | 0.10500 (0.00100) | 0.06042 (0.00252) | 0.75027 (0.00933) | 0.20093 |
| k20 | 1 | 0.55463 (0.01015) | 0.10133 (0.00321) | 0.06048 (0.00417) | 0.74590 (0.00992) | 0.22545 |
| k20 | 2 | 0.55437 (0.01057) | 0.10333 (0.00416) | 0.06092 (0.00397) | 0.74633 (0.01061) | 0.22720 |
| alpha0 | 1 | 0.55027 (0.01194) | 0.10533 (0.00208) | 0.05348 (0.00217) | 0.76913 (0.01108) | 0.17094 |
| alpha0 | 2 | 0.54983 (0.01238) | 0.10333 (0.00289) | 0.05425 (0.00217) | 0.76633 (0.01031) | 0.17147 |
| alpha0.1 | 1 | 0.55259 (0.01091) | 0.10400 (0.00173) | 0.05757 (0.00318) | 0.75763 (0.01148) | 0.19314 |
| alpha0.1 | 2 | 0.55217 (0.01136) | 0.10333 (0.00252) | 0.05792 (0.00374) | 0.75690 (0.00972) | 0.19419 |
| alpha0.4 | 1 | 0.55497 (0.01007) | 0.10267 (0.00473) | 0.06983 (0.00316) | 0.72653 (0.01460) | 0.23721 |
| alpha0.4 | 2 | 0.55448 (0.01016) | 0.10267 (0.00503) | 0.06987 (0.00245) | 0.72543 (0.01398) | 0.23953 |
| lambda0 | 1 | 0.55376 (0.01010) | 0.10400 (0.00346) | 0.06228 (0.00355) | 0.74473 (0.00917) | 0.21159 |
| lambda0 | 2 | 0.55334 (0.01050) | 0.10400 (0.00200) | 0.06317 (0.00317) | 0.74400 (0.00980) | 0.21314 |
| lambda0.025 | 1 | 0.55376 (0.01010) | 0.10467 (0.00306) | 0.06192 (0.00367) | 0.74463 (0.00903) | 0.21159 |
| lambda0.025 | 2 | 0.55334 (0.01050) | 0.10433 (0.00153) | 0.06265 (0.00357) | 0.74490 (0.01028) | 0.21314 |
| lambda0.1 | 1 | 0.55376 (0.01010) | 0.10500 (0.00300) | 0.06075 (0.00347) | 0.74590 (0.00942) | 0.21159 |
| lambda0.1 | 2 | 0.55334 (0.01050) | 0.10400 (0.00200) | 0.06127 (0.00340) | 0.74620 (0.01045) | 0.21314 |
| no_gate | 1 | 0.56170 (0.00584) | 0.09633 (0.00321) | 0.06277 (0.00343) | 0.74623 (0.01058) | 0.18681 |
| no_gate | 2 | 0.56126 (0.00640) | 0.09667 (0.00153) | 0.06310 (0.00349) | 0.74627 (0.00982) | 0.18763 |
| no_historical | 1 | 0.55027 (0.01194) | 0.10533 (0.00208) | 0.05348 (0.00217) | 0.76913 (0.01108) | 0.17094 |
| no_historical | 2 | 0.54984 (0.01223) | 0.10367 (0.00321) | 0.05415 (0.00217) | 0.76650 (0.01041) | 0.17167 |
| no_pair | 1 | 0.55376 (0.01010) | 0.10500 (0.00300) | 0.06127 (0.00365) | 0.74557 (0.00905) | 0.21159 |
| no_pair | 2 | 0.55332 (0.01068) | 0.10433 (0.00153) | 0.06182 (0.00325) | 0.74560 (0.01041) | 0.21291 |
| no_scale | 1 | 0.55101 (0.01180) | 0.10433 (0.00058) | 0.05525 (0.00220) | 0.76593 (0.01170) | 0.18020 |
| no_scale | 2 | 0.55060 (0.01229) | 0.10300 (0.00173) | 0.05607 (0.00210) | 0.76237 (0.00950) | 0.18085 |
| gradient_embedding | 1 | 0.54727 (0.01327) | 0.10067 (0.00115) | 0.05587 (0.00232) | 0.76007 (0.00590) | 0.23214 |
| gradient_embedding | 2 | 0.54656 (0.01368) | 0.09967 (0.00058) | 0.05687 (0.00220) | 0.75737 (0.00491) | 0.23364 |

## Interpretation limits

- Exact uses GPU float32 cosine, HNSW uses CPU normalized-L2 (M=32, efConstruction=80, efSearch=64, four threads). This is a deployment comparison, not a same-device algorithm speed comparison. HNSW insertion can be nondeterministic with multiple threads; all recall measurements are retained.
- RSS includes the Python framework and loaded arrays. Five-millisecond sampling can miss short peaks; allocator retention can produce zero RSS deltas. CUDA measures PyTorch allocated bytes, not all device memory. Timing SD is not scientific seed variation.
- The replay is a random-order pass over final cached scores, not the saved original training trajectory. Seconds divided by ceil(N/8) is amortized replay-batch overhead, NOT measured online graph-update latency.
- No full-training wall time, gradient-cache generation cost, validation loss or accuracy was measured. This experiment cannot establish end-to-end superiority or recover the paper's exact dynamic method.
- The 10k ablation corpus is a subset of the 100k-trained caches; it is not a newly trained 10k model. Gradient embedding is a gradient-space graph, not a second semantic encoder.
- First-order no_pair is intentionally identical to full. Removing history removes the first-order correction entirely. Alpha zero disables correction but retains diversity selection. Zero SD is retained when genuine.
- Pairwise rank-stability correlations share seeds and are dependent; they are descriptive, not independent confidence intervals. No hyperparameters are selected using these outcomes.
