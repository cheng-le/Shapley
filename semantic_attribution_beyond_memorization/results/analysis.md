# Interpretation of the Completed Local Study

Both tasks completed seeds 0/1/2. All 216 query ranks were recomputed from saved
scores; all 10,404 domain records passed step/count/normalization checks. Every
dynamics document was visited, and every paraphrase document was visited once.

## Paraphrase Rankings

Graph-only correction improves mean designated-source rank from 415.33 to
335.78 for paraphrases, from 616.67 to 554.44 for significant paraphrases, and
from 694.00 to 660.78 for similar-topic queries. Partial-exact rank worsens from
354.22 to 360.89. These are category means across three fixed cases and three
training seeds, not independent evidence from a large query benchmark.

BM25 and embedding retrieval substantially outperform both local attribution
variants. Thus the study supports a limited relative improvement from graph
correction over its pointwise inputs; it does not demonstrate state-of-the-art
semantic retrieval or independence from pretraining memorization. Similar-topic
queries have only a designated comparison source, not uniquely correct provenance.

First- and second-order ranks coincide in every category. The saved scores are
not identical: maximum absolute differences are approximately 1.05e-8, 7.96e-9,
and 7.58e-9 across the three seeds. The inherited local curvature surrogate is
too small to change the measured ranks. Do not count these as two independent
confirmations of a strong second-order effect.

## Domain Dynamics

The final mathematical-domain share of positive totals increases from 25.103%
to 26.726% after graph correction. Its signed total decreases from approximately
1.733e-3 to 1.652e-3. Higher normalized share therefore does not mean higher
absolute mathematical attribution or improved validation performance.

Inspect domain_history.csv for signed totals and seen-item means for all domains.
Composition plots average normalized shares over seeds and clip negative domain
totals only for visualization. Contributions are sensitive to unequal domain
sample counts, repeated visits, and the local score approximation.

This study uses accumulated-score graph regularization, not the full draft's
online Adam-aware algorithm. It is suitable as an explicitly labeled local
case study; the limitations should accompany any paper presentation.
