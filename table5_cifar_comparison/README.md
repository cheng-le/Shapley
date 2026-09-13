# CIFAR10 Table 5 comparison

This directory implements the data-selection experiment from Table 5 of
`Data Shapley in One Training Run` using the local CIFAR10 files under
`data/CIFAR10/cifar-10-batches-py`.

## Protocol

The default arguments follow Appendix E.6:

- Three independently sampled CIFAR10 training subsets of size 1,000.
- Exactly 10% of each subset receives a uniformly chosen incorrect label.
- ImageNet-pretrained ResNet18.
- Adam, learning rate `1e-3`, weight decay `1e-4`, label smoothing `0.1`.
- 50 epochs, batch size 64, and a 0.1 learning-rate multiplier every 10 epochs.
- High-value subsets at 20%, 40%, 60%, and 80% are retrained and evaluated.
- Data Shapley, Datamodels, and Empirical Influence share 1,000 random-coalition
  training runs per corrupted dataset.
- TRAK uses projection dimension 2,048 and reports 1/5/15/25 checkpoint averages.
- LESS uses epochs 10/20/30/40/50 and projection dimension 2,048.
- KNN-Shapley uses `K=5` and ResNet18 penultimate-layer features.
- The paper's final two In-Run rows follow an Adam training trajectory but use
  the paper's first- and second-order SGD update proxy for attribution. The
  paper explicitly notes that its efficient formulation is tailored to SGD.
- The extended rows add explicit Adam-aware and robust graph Adam-aware scores.

Run the complete experiment with:

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python table5_cifar_comparison/run_table5.py
```

This is intentionally a long experiment: the paper protocol includes thousands
of ResNet18 fits. Results are cached after every scoring method and every
selection evaluation. Re-running the command resumes from those caches.

Use the end-to-end smoke-test profile with:

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python table5_cifar_comparison/run_table5.py --quick
```

Individual scorers can also be run directly, for example:

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python table5_cifar_comparison/knn_shapley.py --quick
```

## Implementation boundaries

KNN-Shapley uses the standard exact recurrence on learned features. Datamodels
uses ridge regression over subset-inclusion indicators. Empirical Influence is
the included-versus-excluded utility difference. Retraining Data Shapley uses
the Shapley-kernel weighted regression estimator over coalition utilities.

The repository does not include the official TRAK/LESS packages or the paper's
architecture-specific ghost-gradient kernels. TRAK and LESS are therefore
implemented with projected last-layer loss gradients; influence functions use
a conjugate-gradient inverse empirical Fisher in the same last-layer space;
first- and second-order In-Run Data Shapley use last-layer ghost gradients, with
the second-order method applying an empirical-Fisher HVP. These variants obey
the Table 5 data/training/selection protocol but should not be described as
bitwise reproductions of the original authors' external implementations.

The Graph rows reuse the exact cached Adam-aware score vectors and apply one
deterministic conservative update on a 10-nearest-neighbor semantic graph. The
graph uses ImageNet-pretrained ResNet18 embeddings extracted after resizing
CIFAR images to 128x128. Only neighbors with the same observed label provide
score support. Propagation can lower unsupported high scores but can never raise
a point above its raw score, preventing a mislabeled low-value point from
inheriting high value. A scale-aware penalty further downweights examples whose
observed label disagrees with their semantic neighborhood. No clean labels or
test-set outcomes are used by this correction.

Outputs are written to `table5_cifar_comparison/results/`:

- `table5_results.json`: protocol, per-run accuracies, means, and deviations.
- `table5_results.csv`: machine-readable long-form statistics.
- `table5_results.md`: formatted final table.
- `scores/`: resumable attribution-score caches.
- `retraining/`: shared random-coalition utilities.

