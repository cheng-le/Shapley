# Semantic Attribution Beyond Memorization

Section 4.5 of PROJECT_MIGRATION_PROMPT.md: two complementary local-scale
studies, following high-value and diverse data selection.

## Questions

1. How does the rank of a designated source change from partially verbatim
   queries to paraphrases and related-topic queries?
2. How do signed domain contributions evolve along an actual training run
   whose utility queries come from DM Mathematics?

## Fixed Settings

- GPT-2 124M pretrained checkpoint; only final LayerNorm's 1,536 parameters
  are updated and differentiated. Sequence length 128, batch size 8.
- AdamW, betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0;
  learning rate warms up for 20 steps to 6e-4, then decays to 6e-5.
- Training seeds 0/1/2; fixed data sampling seed 2027.
- Graph: 10 nearest neighbors, nonnegative row-normalized cosine weights,
  alpha 0.2 and disagreement penalty 0.05, inherited from the case-study code.
- Same initial encoder for document and query embeddings. BM25 uses the same
  visible 128-token window as the neural methods.

### Paraphrase Ranking

2,048 candidates: three designated sources from the existing local case file
and 2,045 sampled Pile training documents. Twelve fixed queries cover three
source groups and four categories: partial exact, paraphrase, significant
paraphrase, and similar topic. These are the repository's existing case texts;
we have not independently established verbatim transcription of the reference
paper. Each seed trains one pass (256 updates).

Compare local first/second-order In-Run, the corresponding graph-only score
corrections, local BM25, and embedding cosine. Lexical/semantic fusion is not
included in the graph rows, so retrieval alone cannot account for a graph row's
performance. Report per-query ranks, category mean rank, MRR, and Hit@10.
Category statistics average the three cases within each seed, then report
mean/population std across seeds. Related-topic queries do not have a unique
gold provenance source: their designated-source ranks are qualitative only.

### Domain Dynamics

Reservoir sample up to 128 documents per available training domain: 1,817
documents across 17 domains. Rare domains contribute all available documents.
Use 16 mathematical documents sampled from the authoritative validation file;
verify raw-document train/validation hashes do not overlap. Run 500 updates
per seed, preserving short final batches. Record steps 1,10,...,500.

Compare pointwise and graph-corrected first/second-order scores on the same
training trajectory. Save signed domain sums, means over seen items, positive
contribution shares, per-document visits, and all training losses. Positive
shares clip negative domain totals only for the composition plot; signed sums
remain in JSON/CSV. Domain sums depend on sampling sizes and visitation.

## Interpretation Boundary

This is an expanded local-scale case study, not the reference paper's
320K-document/10B-token experiment and not a main 100K-corpus run.
The implementation inherits the case-study SGD update proxy along AdamW and
a capped diagonal-Fisher second-order surrogate. Graph correction operates on
accumulated scores; it is not the draft's online Adam-aware graph algorithm or
the main-table Graph v2 replay. Results measure this explicit local variant.
Only three hand-picked source groups are available, so good rankings would not
establish broad semantic generalization or exclude pretraining memorization.
All methods and categories are retained, including unfavorable outcomes.

## Reproduction

```bash
python -m semantic_attribution_beyond_memorization.prepare_data
python -m semantic_attribution_beyond_memorization.run_all
python -m semantic_attribution_beyond_memorization.aggregate
```

run_all uses GPUs 0/1 and writes isolated task/seed logs and results. Source
experiment outputs are not overwritten. Inputs, all query ranks, score arrays,
signed trajectories, aggregate tables and PNG/PDF/SVG figures are retained.
