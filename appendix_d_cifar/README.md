# Appendix D: Ablations and Sensitivity

This independent directory completes the Appendix-D diagnostic design in
`PROJECT_MIGRATION_PROMPT.md`. It does not overwrite the Pile or CIFAR Table-5
experiments. Pile's existing weak noise-detection and subset-retraining results
remain reported in `efficiency_scalability_ablations/second_order_online/`.

## Scope and Data

Use the official local CIFAR-10 files, an ImageNet-pretrained ResNet18 encoder,
and a trainable linear classification head. The encoder is FROZEN. The head
uses normalized 512-dimensional features plus a bias coordinate (5130 trained
weights). This is not full-network Table-5 training or its old conservative,
observed-label graph correction. Inputs use the existing ImageNet normalization.
The official ResNet18 checkpoint was downloaded to this directory and its
SHA-256 prefix was verified. Its full hash is recorded in the data manifest.

Three data seeds each sample 1000 training images, with exactly 100 labels
changed to a different class. A fixed set of 1000 other TRAIN images is reserved
for validation; its first 128 supply attribution queries. These indices are
excluded from every training subset. The final evaluation uses 2000 fixed
official TEST images. No final-test labels enter features, graph construction,
gating, scoring or selection. Unlike the original Table-5 context helper, this
protocol does not draw both validation and evaluation from the same test pool.

## Trajectories and Selection

Data seeds 0/1/2 are crossed with training seeds 0/1/2: nine actual trajectories.
Each trains the linear head for five epochs with Adam (lr=.001,
weight_decay=.0001, batch=64, label_smoothing=.1). Per-example softmax gradients
are computed analytically and tested against PyTorch autograd. First-order
sample-conditioned Adam scores are recorded with the actual batch IDs and
epoch boundaries. No HVP or second-order score cache is used.

All graph configurations replay those same observed contributions in their
actual order, accumulating historical RAW scores across epochs. Graph scores
do not affect model updates. The optional gate uses positive agreement of
gradients evaluated at the INITIAL head, before training, not gradients from
future steps. This is a static initial-gradient gate, not a claimed exact
dynamic-gate reproduction. `full` includes this optional gate; `no_gate`
corresponds to the semantic-only propagation weights in the current TeX.

Every selection retains 200 examples (20%). Each distinct selected set is
genuinely retrained from the same paired head initialization for 50 epochs,
using the observed, potentially corrupted labels. Adam settings match scoring,
with StepLR multiplying lr by .1 every 10 epochs. The encoder remains frozen.
Within a data/training seed pair, identical selected sets reuse the same fit;
`utility_reused` and `unique_fit_id` expose this rather than count duplicates
as independent training. Classifier features always remain ResNet128.

Random and full-data references were added after the initial low-accuracy
diagnostic without changing any original sweep hyperparameters. Random uses
unpenalized Top-k of independent random scores; graph settings in its shared
configuration template are ignored. The full-data reference trains on all 1000
examples for 50 epochs and thus has more optimizer updates than a 200-example fit.

## Fixed Configuration Grid

- k = 5, 10, 20; alpha = 0, .2, .4; lambda_fraction = 0, .05, .1: 27 cells.
- Separately remove gate, history, pair, and scale matching; removing diversity
  is already in the lambda=0 grid. Defaults: k=10, alpha=.2, lambda=.05.
- Alternative graph embeddings: ResNet at 32x32 instead of 128x128, and
  normalized 8x8 average-pooled pixels. These change the graph, not the head.
- Propagation adjacency: directed, arithmetic-mean symmetric, mutual-only
  symmetric. Symmetric propagation weights are not renormalized again.
  The selection penalty always uses symmetric affinity, as in the paper.
- HNSW: M=4, efConstruction=20, efSearch=2/8/32/128. Recall@10 is measured
  against exact neighbors on the same representation, not inferred from ef.
- HNSW efSearch=8 uses graph seeds 0/1/2 controlling insertion permutation,
  with one FAISS thread for reproducibility. This gives a fully crossed
  3 data x 3 training x 3 graph seed design (27 cells).
- Exact graphs also use nominal seeds 0/1/2 as deterministic negative controls.

In total, 43 graph configurations plus random x 9 trajectories = 396 result
cells, backed by 311 distinct selected-set fits and 9 full-data reference fits. All grids
were fixed before results were inspected. No winning configuration is selected
on the final test set. The default is not relabeled as an optimized winner.

## Metrics and Interpretation

Report test accuracy/loss after actual subset retraining, corruption AUROC
using negative attribution, selected corruption fraction, and diversity on
one fixed semantic k=10 reference graph per dataset. Binary one-hop coverage
and pair affinity use only positive edges, excluding zero-padding artifacts.

Mean/sample SD over the nine crossed cells is descriptive, not nine iid
dataset replications. Training-seed rank stability is computed only on the
same dataset IDs. Data-seed subsets can overlap. Graph seed varies insertion
only, leaving data/labels and training fixed; exact seed invariance is checked.
Genuine zero variability is preserved. No significance or equivalence claims
are made from these few seed levels.

## Running and Outputs

```bash
/home/lcheng/miniconda3/envs/esm_test/bin/python appendix_d_cifar/run.py all
/home/lcheng/miniconda3/envs/esm_test/bin/python appendix_d_cifar/audit.py
/home/lcheng/miniconda3/envs/esm_test/bin/python appendix_d_cifar/diagnose.py
```

The script uses GPUs 0/1/2 for dataset workers and GPU3 for feature extraction.
The isolated FAISS installation from the preceding efficiency experiment is
reused. Completed dataset/training-seed groups resume from their marker; a
partially completed group is rerun as a unit. Other directories are read-only.

`data/manifest.json` contains all dataset indices and feature hashes.
`results/protocol.json` contains the fixed grid and source hash.
Each `results/data*_train*/` saves raw trajectories, scores, selected sets,
test predictions and per-test-example losses. Complete tables, seed-factor
data, training rank stability and reports are in `results/`.
Four figure groups are exported as PNG/PDF/SVG with their CSV source data:
parameter heatmaps, module/embedding/symmetry ablations, ANN recall versus
quality, and separated seed sensitivity.

See [the experimental analysis](EXPERIMENT_REPORT.md) for the full interpretation,
including the default Graph's lower subset accuracy than random and the class
imbalance diagnosis. These frozen-head results do not establish a full-network
utility improvement.
