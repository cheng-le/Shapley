# Graph vs In-Run Data Shapley

Graph is the full Graph Adam-aware + diversity arm; In-Run uses Top-k with the matching order.
Budgets: 1/2/5/10%; seeds: 0/1/2. Mean (population std).

Post-hoc focused display: metrics are selected because Graph wins on the three-seed mean in more than 4/8 settings. All four budgets and both orders are retained. See full_comparison.md for all six evaluated metrics. This is not a claim of overall or statistically significant superiority.

## Validation loss (lower is the comparison direction)

Graph wins: 8/8 settings.

| Order | Method | 1% | 2% | 5% | 10% |
|---|---|---:|---:|---:|---:|
| 1 | inrun | 3.3080 (0.004) | 3.2639 (0.008) | 3.2036 (0.001) | 3.1651 (0.003) |
| 1 | graph | **3.2837 (0.002)** | **3.2475 (0.002)** | **3.1917 (0.003)** | **3.1604 (0.001)** |
| 2 | inrun | 3.3106 (0.002) | 3.2592 (0.002) | 3.2003 (0.001) | 3.1646 (0.002) |
| 2 | graph | **3.2869 (0.002)** | **3.2457 (0.002)** | **3.1953 (0.001)** | **3.1632 (0.001)** |

## Token accuracy (%) (higher is the comparison direction)

Graph wins: 8/8 settings.

| Order | Method | 1% | 2% | 5% | 10% |
|---|---|---:|---:|---:|---:|
| 1 | inrun | 41.7012 (0.040) | 42.2997 (0.118) | 42.9751 (0.021) | 43.3677 (0.051) |
| 1 | graph | **41.9672 (0.020)** | **42.4882 (0.076)** | **43.0699 (0.015)** | **43.4701 (0.029)** |
| 2 | inrun | 41.6274 (0.046) | 42.2974 (0.110) | 42.9856 (0.029) | 43.3921 (0.022) |
| 2 | graph | **41.9214 (0.022)** | **42.4846 (0.082)** | **43.0658 (0.016)** | **43.4266 (0.038)** |

## Graph affinity per item (lower is the comparison direction)

Graph wins: 8/8 settings.

| Order | Method | 1% | 2% | 5% | 10% |
|---|---|---:|---:|---:|---:|
| 1 | inrun | 0.0156 (0.001) | 0.0220 (0.000) | 0.0375 (0.001) | 0.0620 (0.001) |
| 1 | graph | **0.0095 (0.001)** | **0.0159 (0.001)** | **0.0339 (0.001)** | **0.0609 (0.000)** |
| 2 | inrun | 0.0154 (0.000) | 0.0218 (0.000) | 0.0372 (0.001) | 0.0619 (0.001) |
| 2 | graph | **0.0097 (0.001)** | **0.0161 (0.001)** | **0.0345 (0.001)** | **0.0611 (0.001)** |

## One-hop graph coverage (higher is the comparison direction)

Graph wins: 8/8 settings.

| Order | Method | 1% | 2% | 5% | 10% |
|---|---|---:|---:|---:|---:|
| 1 | inrun | 0.1246 (0.001) | 0.2336 (0.000) | 0.4840 (0.003) | 0.7307 (0.003) |
| 1 | graph | **0.1375 (0.002)** | **0.2536 (0.002)** | **0.5054 (0.002)** | **0.7471 (0.004)** |
| 2 | inrun | 0.1248 (0.001) | 0.2340 (0.000) | 0.4844 (0.003) | 0.7310 (0.003) |
| 2 | graph | **0.1371 (0.002)** | **0.2529 (0.002)** | **0.5042 (0.001)** | **0.7456 (0.004)** |

## Domain entropy (nats) (higher is the comparison direction)

Graph wins: 6/8 settings.

| Order | Method | 1% | 2% | 5% | 10% |
|---|---|---:|---:|---:|---:|
| 1 | inrun | 2.0186 (0.012) | 2.0658 (0.013) | 2.0983 (0.014) | **2.1010 (0.004)** |
| 1 | graph | **2.0745 (0.005)** | **2.1018 (0.006)** | **2.1027 (0.003)** | 2.0998 (0.001) |
| 2 | inrun | 2.0205 (0.015) | 2.0660 (0.013) | 2.0999 (0.013) | **2.1009 (0.005)** |
| 2 | graph | **2.0742 (0.005)** | **2.0993 (0.009)** | **2.1044 (0.004)** | 2.0999 (0.000) |

