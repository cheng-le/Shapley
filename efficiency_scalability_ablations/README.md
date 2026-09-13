# Efficiency, Scalability, and Ablations

This directory implements the next subsection (4.6) of
`PROJECT_MIGRATION_PROMPT.md` without modifying previous experiments.

Completed: 12 timing cells, 168 ablation cells, consistency checks and three
plot formats. See [the Chinese experimental report](EXPERIMENT_REPORT.md).
Those original measurements exclude training and genuine online updates.
The separate [HVP versus Hessian-free online follow-up](second_order_online/EXPERIMENT_REPORT.md)
now measures matched 1k training trajectories, real HVP costs and subset
retraining quality. Its measured speedups are not 100k end-to-end measurements.

## Purpose and Protocol

Measure sparse graph construction, gradient gating, cached-score replay and
diversity selection at 1k, 10k, 50k and 100k corpora. Compare the existing GPU
chunked exact cosine KNN with CPU FAISS HNSW and a pointwise score-replay
baseline. This is a component benchmark, not end-to-end attribution timing.

Inputs are GPT-2/Pile caches from `pile_main_100k/cache`: semantic embeddings
(768 dimensions), gradient features (1536 dimensions), and Adam-aware scores.
Training is not repeated. The source protocol uses pretrained GPT-2, freezes
all parameters except the final LayerNorm (1536 trainable parameters), truncates
to 128 tokens and performs one AdamW pass (batch=8, lr=6e-4, betas=0.9/0.95,
weight decay=0.1). Semantic features are pooled before that training pass;
gradient features and pointwise contributions are collected along the pass.
This is not full-parameter GPT-2 fine-tuning. A fixed seed-2027 permutation defines nested subsets.
Timing uses the clean seed-0 cache, three separate worker processes per scale,
GPU 2 (RTX 3090), four CPU threads, float32 exact KNN, k=10 and chunks of 512.
HNSW uses normalized L2, M=32, efConstruction=80, efSearch=64, and four CPU
threads. Exact/ANN build order alternates across repetitions. Input loading and
GPU warmup are excluded. ANN recall and downstream agreement are measured,
not assumed. CPU/GPU times are deployment costs on different hardware.

Ablations use all three cache seeds (0, 1, 2), clean and corrupted scenarios,
the same 10k subset, and 10% selection. This subset does not represent a model
trained on only 10k examples. Both first and second order are retained.
Default: k=10, alpha=0.2, lambda_fraction=0.05, with gating, historical replay,
scale matching and second-order pair correction. One-factor-at-a-time variants:

- k: 5, 10, 20.
- alpha: 0, 0.1, 0.2, 0.4.
- lambda_fraction: 0, 0.025, 0.05, 0.1; actual penalty is this times score SD.
- Remove gate, history, pair or scale matching separately.
- Replace semantic embeddings with normalized gradient-space features.

Noise AUROC uses negative contribution as a corruption score. Selected noise
fraction is lower-is-better. Pair affinity per selected sample and binary
one-hop coverage are evaluated on a fixed semantic k=10 reference graph per
seed, including for the gradient-embedding and k variants. Entropy uses natural
logarithms. Rank stability is mean pairwise Spearman over clean seed scores;
these three correlations are dependent. Means and sample SDs are preserved
without inventing nonzero variation. No outcome-driven tuning or filtering.

## Running

FAISS 1.7.4 is installed only under this directory's `vendor/`.

```bash
/home/lcheng/miniconda3/envs/esm_test/bin/python efficiency_scalability_ablations/study.py all
/home/lcheng/miniconda3/envs/esm_test/bin/python efficiency_scalability_ablations/analyze.py
```

Completed JSON cells are reused on restart. Delete only the corresponding
new experiment cell before intentionally rerunning a configuration. Existing
main-study caches and results are read-only. `test` reruns consistency checks;
`summarize` rebuilds reports and plots from stored raw measurements.

## Artifacts and Limits

- `results/timing/`: raw stage time, CPU RSS and CUDA allocation measurements,
  ANN recall, score agreement, selected-set overlap and payload sizes.
- `results/ablation/`: all seed-level diagnostics and clean score vectors.
- `results/*_all.csv`, `*_summary.csv`: complete raw and aggregate tables.
- `results/report.md`: full report, including unfavorable or null outcomes.
- `results/resources.md`, `pipeline_all.csv`: combined cached-stage costs,
  amortized replay-batch time, CPU/GPU memory and ANN fidelity/storage.
- `results/input_subset_summary.json`: cache fingerprints and subset composition.
- `results/efficiency_ablation.{png,pdf,svg}`: compact overview.
- `results/tests.json`, `provenance.json`: validation and source/environment metadata.

CPU RSS is sampled every 5 ms, with absolute peaks and deltas; brief peaks may
be missed and allocator retention can give zero deltas. CUDA figures measure
PyTorch allocated bytes, not device-wide usage. Cache sizes distinguish sparse
array payload, input tensors and serialized ANN index; payload is not compressed
disk usage. Timing repetitions do not constitute independent training seeds.

The existing Graph-v2 replay visits final cached contributions in a random
order. It does not reproduce the saved original training trajectory. Replay
time per batch is therefore not actual online graph-update latency. Full
gradient extraction, training wall time, online dynamic updates and downstream
retraining quality are NOT measured here and remain limitations of subsection
4.6. This study must not be described as a complete end-to-end paper reproduction.
