# Requested Reporting Version

## Paraphrase Ranking (Legacy Fusion Case Study)

This table uses 512 candidates and seed 0. Each entry averages the designated-source ranks over three query groups. It is not a three-seed result or the newer 2,048-candidate graph-only experiment.

| Method | Partial exactly the same | Paraphrase | Significant paraphrase | Similar topic |
|---|---:|---:|---:|---:|
| 1st Order In-Run Data Shapley | 141.7 | 144.0 | 136.0 | 146.7 |
| 2nd Order In-Run Data Shapley | 141.7 | 144.0 | 136.0 | 146.7 |
| Influence Function | 1.0 | 1.0 | 5.7 | 26.0 |
| BM25 | 1.0 | 2.7 | 2.3 | 231.3 |
| Graph 1st Order In-Run (ours) | 1.0 | 1.0 | 1.3 | 26.7 |
| Graph 2nd Order In-Run (ours) | 1.0 | 1.0 | 1.3 | 26.7 |

Suggested reporting text:

In the 512-candidate local case study, the graph-based fusion variants achieve mean source ranks of 1.0, 1.0, 1.3, and 26.7 for partial-exact, paraphrased, significantly paraphrased, and similar-topic queries, respectively. They tie for the best ranks in the first two categories and obtain the best rank under significant paraphrasing. For similar-topic queries, their rank of 26.7 is close to the local influence-function baseline (26.0) and better than BM25 (231.3). The graph rows combine semantic and lexical retrieval with graph-regularized in-run scores; these results evaluate that fusion system and do not isolate the effect of graph propagation.

The newer graph-only result remains in summary.md. The two settings must not be merged as one controlled experiment.

## Latest Domain-Dynamics Figure

1,817 training documents, 17 domains, 16 separate math validation documents; seeds 0/1/2 and 500 updates per seed. Training inputs are truncated to 128 tokens.

| Domain | Training documents |
|---|---:|
| ArXiv | 128 |
| DM Mathematics | 128 |
| Enron Emails | 128 |
| EuroParl | 69 |
| FreeLaw | 128 |
| Github | 128 |
| Gutenberg (PG-19) | 41 |
| HackerNews | 128 |
| NIH ExPorter | 128 |
| PhilPapers | 32 |
| Pile-CC | 128 |
| PubMed Abstracts | 128 |
| PubMed Central | 128 |
| StackExchange | 128 |
| USPTO Backgrounds | 128 |
| Ubuntu IRC | 11 |
| Wikipedia (en) | 128 |

The legacy figure3_pile_graph_dynamics result used 32 documents per domain (544 total) and four math validation documents. It is a separate dataset snapshot and protocol.

## Reproducibly Selected Examples

Highest/lowest refer to the mean final Graph second-order local score over seeds 0/1/2. They are extrema, not estimates of the typical document in each domain. The validation exemplar maximizes mean cosine to the other validation documents using frozen initial GPT-2 representations.

### highest: DM Mathematics, index 193

SHA256: `4d0be424b3945cbd4bfa44f5a1114bc71066c3a80aad55c51af861cb5bb078ab`

Mean final score: +7.015670e-05.

Visible input window:

```text
-213*x**2 - 10*x - 1
What is the z'th term of 169, 326, 483, 640, 797, 954?
157*z + 12
What is the o'th term of -96, -169, -218, -243, -244?
12*o**2 - 109*o + 1
What is the a'th term of 8, 16, 36, 74, 136?
a**3 + a + 6
What is the f'th term of 16, 39, 100, 217, 408, 691?

```

### lowest: Github, index 584

SHA256: `9ddff0d24fa129761797c531aa2c18cc83e7f3d8634d87d7ff60f03f24c2e864`

Mean final score: -8.621297e-05.

Visible input window:

```text
!COMPULSORY
!INCREMENTAL

<option keepgoing>
<option tools>
<option epoconly>

e32tools	e32tools\group
e32			e32
compsupp	e32\compsupp
f32			f32\group
hal			hal
```

### validation_medoid: DM Mathematics, index 12

SHA256: `01cf5a64f003a2acf6caa7ca609bbe7dcd4cb25e2fd409a1878ccc9fb728c01d`

Visible input window:

```text
34 = -4*i + 3*z, 8*i = 2*z + 3*z + 3502 - 3564 for i.
-4
Solve -2*n = -2*p - 18, 94*p - 12*n - 20 = 86*p + 12*p for p.
-8
Solve -6*k - 11*q + 13 = 117, -2*k + 46 = -k - 6*k + q for k.
-10
Solve 42 = 5*s - 67*w, 286*w - 11 =
```

A low score is relative to the mathematical utility objective; it does not imply that the original document is generally low quality.
