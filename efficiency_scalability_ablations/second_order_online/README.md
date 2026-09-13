# Hessian-Free Online Graph vs HVP

This independent experiment tests the paper's actual second-order efficiency
claim. It does not reuse or relabel previous Graph-v2 scores, which already
contain Adam-aware HVP corrections. Existing experiments remain unchanged.

## Protocol

- GPT-2, only final LayerNorm trainable (1536 parameters), 128 tokens.
- 1000 fixed Pile corpora; AdamW lr=6e-4, betas=.9/.95, weight decay=.1,
  gradient clipping=1, batch=8, one pass of 125 updates.
- Three seeds, each on its own RTX 3090 with two CPU threads; method order
  rotates within seed. Each method/scenario runs in its own process.
- Four methods: SGD-style In-Run second order, Adam-aware second order,
  first-order Adam-aware plus online Graph surrogate, first-order Adam-aware.
- The two second-order baselines each use ONE real HVP per batch, not a dense
  Hessian or an HVP per example. The original 25% curvature-SD cap is retained.
- Graph follows the TeX semantic history/pair/scale equations (k=10, alpha=.2).
  It has no optional gradient gate, no cached second-order score, no access
  to future gradients, and no higher-order autograd requests.
- Clean and 10%-corrupted scenarios: 24 trajectories and 3000 measured updates.
  Corruption uses the existing `common.corrupt_rows` with seed+991.
- Every method uses identical batches, dropout RNG and optimizer updates.
  Final parameters, all training losses and common first-order Adam scores
  passed bitwise equality checks across methods within each seed/scenario.

## Timing

Graph embeddings are freshly extracted on each scenario's actual inputs, and
exact KNN is built locally. Both costs are included in reported Graph totals.
Two warmup updates precede each measurement, then model LayerNorm, optimizer,
RNG and history are reset. Checkpoint loading/tokenization and warmup are
excluded consistently. No graph preparation is hidden in an earlier cache.

CUDA-synchronized stages separate differentiable query preparation, sample
gradients, directions, first-order scoring, HVP or graph interaction, output
transfer and the optimizer update. Graph interaction includes CPU transfer,
diffusion and pair aggregation. HVP interaction excludes differentiable query
preparation, but that cost IS included in attribution and pipeline totals.

Attribution totals exclude optimizer updates. Pipeline wall time includes
training and attribution, plus Graph embedding/KNN preparation. Raw files also
contain per-step timings, online/preparation CUDA peaks, sampled process RSS,
autograd counters and in-batch graph-edge counts. CUDA metrics are allocated
PyTorch bytes, not device-wide memory; RSS sampling at 5 ms can miss peaks.

## Quality

Noise AUROC comes from separate corrupted trajectories. Each clean method's
Top-10% subset is retrained from the same GPT-2 initialization for one pass:
100 corpora, 13 updates. The 128 evaluation examples are disjoint from the
16 attribution-utility examples. Graph with lambda=.05 diversity is a separate
fifth selection arm. All 15 subset retraining cells are retained, including
unfavorable results. Validation loss averages per-example token-averaged NLL.

## Reproduction and Artifacts

```bash
/home/lcheng/miniconda3/envs/esm_test/bin/python efficiency_scalability_ablations/second_order_online/run.py all
/home/lcheng/miniconda3/envs/esm_test/bin/python efficiency_scalability_ablations/second_order_online/audit.py
```

`data/manifest.json` records source indices, corruption and input hashes.
`results/provenance.json` records code hashes and environment. Raw timing,
score vectors, training losses and parameters are in `results/cells/`;
`results/quality/` contains retraining results and selected indices.
`results/report.md` and CSV files provide all results; PNG/PDF/SVG plots are
available. Completed cells are reused on restart. Delete only a specific new
cell when intentionally rerunning it.

Tests check analytic HVP correctness, online/replay equality, no future-history
access, zero-alpha identity and a runtime guard against higher-order Graph
autograd. Actual trajectories also check zero Graph HVP calls.

This is a real 1k online benchmark, not a full 100k trajectory or full-parameter
fine-tuning. Earlier four-scale cached results remain separate. A graph
surrogate is not mathematically equivalent to an HVP. A large interaction-only
speedup must not be described as the same end-to-end speedup.
