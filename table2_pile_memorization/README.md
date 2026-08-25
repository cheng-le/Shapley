# Table 2: paraphrase robustness

This folder reproduces the ranking protocol in Table 2 of *Data Shapley in One
Training Run*. The three source/query groups are transcribed from Appendix E.5.
It compares first/second-order In-Run values, a diagonal-Fisher influence
function, BM25 (`k1=1.5`, `b=0.75`), and query-anchored graph-aware
first/second-order values. Fusion weights do not use target labels: confident
lexical anchors are combined with the semantic view; low-confidence queries
retain graph-smoothed In-Run signal only when semantic and lexical Top-10
neighborhoods are completely disjoint.

The paper ranks against about 320k corpora from a 10B-token Pile subset. The
local machine has a 192MB, 17-domain `pile-uncopyrighted` extraction, so the
default executable profile ranks against 512 corpora. GPT2-124M and the paper's
AdamW hyperparameters are retained, while only the final LayerNorm is updated
and differentiated. Both target and local parameters are recorded in JSON.

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python table2_pile_memorization/run_table2.py
```
