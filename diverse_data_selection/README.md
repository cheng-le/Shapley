# High-Value and Diverse Data Selection

This is the next experiment group after robustness and attribution stability,
corresponding to Section 4.4 of PROJECT_MIGRATION_PROMPT.md.

## Purpose

Test whether score correction and redundancy-aware selection contribute
independently to useful, diverse subsets. Compare the following four arms for
both first- and second-order cached scores:

- Adam Top-k
- Graph Top-k
- Adam + diversity
- Graph + diversity

## Fixed Protocol

Pile: 100,000 training documents from monology/pile-uncopyrighted, GPT-2 124M,
128-token truncation, final-LayerNorm attribution, seeds 0/1/2. Budgets are
1/2/5/10 percent, following the most recent low-ratio experiment. The historical
20/40/60/80 percent budgets can be requested separately with --budgets.

All arms use the same frozen semantic graph within each seed. Diversity uses
the existing symmetric-graph greedy selector, with lambda equal to 0.05 times
the standard deviation of the input scores. Graph scores are the frozen v2
cache used by the main experiment; this is a cached-score replay approximation,
not a newly recorded online trajectory. No parameters are tuned in this study.

Metrics: induced graph affinity per selected item (lower is less redundant),
one-hop graph coverage, domain entropy in nats, number of represented domains,
nonpadding input tokens, next-token training targets, and selected-set overlap
with the corresponding Top-k arm. Graph affinity uses row-normalized weights;
it is not a raw mean cosine similarity or a duplicate rate.

For downstream utility, existing cells are reused only after validating the
seed, budget, method, selection strategy, selected-index hash, evaluation split,
epoch count, and learning-rate schedule. Validation loss is document-averaged
cross entropy; exp(loss) is explicitly a document-weighted perplexity proxy.
Missing arm evaluations remain missing, not inferred from another arm.

CIFAR-10 Table 5 provides a separate cross-modal reference: 1,000 images per
seed, ImageNet-pretrained ResNet-18, and 20/40/60/80 percent selection. Those
existing runs do not supply all four module ablations, so they are not merged
into the Pile ablation table.

## Run

Use a Python environment with NumPy, SciPy, PyTorch and Transformers:

```bash
python -m diverse_data_selection.run_experiment
python -m diverse_data_selection.run_training --gpus 0 1
python -m diverse_data_selection.run_experiment
python -m diverse_data_selection.plot_results
```

The analysis runner saves exact selections and imports validated utility cells.
results/pending_evaluations.json lists missing downstream evaluations.
The training runner evaluates them with full-parameter GPT-2, AdamW,
learning rate 6e-5, betas (0.9, 0.95), weight decay 0.1, batch size 16,
one epoch, gradient clipping at 1.0, and constant learning rate. It uses the
evaluation partition of protocol_v2_split.json and resumes completed cells.
Re-run the analysis after training to refresh the table.

Outputs include metrics.csv, summary.md, summary.json, selected/*.npy and a
source manifest. All source caches and previous results remain read-only.

## Comparison With In-Run Data Shapley

```bash
python -m diverse_data_selection.compare_inrun
```

This compares full Graph Adam-aware + diversity with the corresponding first-
or second-order In-Run Top-k baseline under the same low-ratio protocol.
results/inrun_comparison/full_comparison.md retains all six evaluated metrics.
focused_comparison.md is an explicitly post-hoc display of metrics for which
Graph has a strictly better three-seed mean in more than four of the eight
order-budget settings. No budgets or seeds are removed. Win counts do not
establish statistical significance, and this comparison alone does not isolate
graph correction from Adam awareness; use the four-arm ablation for that purpose.
