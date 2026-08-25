# Figure 3: graph-aware domain dynamics

This folder recreates Figure 3's domain-contribution visualization using only
our graph-aware second-order In-Run method. Validation corpora come exclusively
from `DM Mathematics`. At each recorded step, negative domain totals are
excluded and the remaining totals are normalized to 100%, matching the paper.

The paper tracks 10,000 updates over a 10B-token, 22-domain Pile sample. Local
data contains 17 uncopyrighted domains, so the executable default uses 32
corpora per available domain and 500 updates. GPT2-124M and paper AdamW settings
are retained, with attribution/update gradients restricted to final LayerNorm.

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python figure3_pile_graph_dynamics/run_figure3.py
```
