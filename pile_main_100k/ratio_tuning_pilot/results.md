# Low-ratio independent tuning pilot

Legacy v2 scores; 1,999-row valuation split only; final 8,001-row evaluation split untouched.

| Ratio | Random | Adam | Graph | Graph-Adam (pp) | Graph-Random (pp) |
|---:|---:|---:|---:|---:|---:|
| 5% | 0.431401 (0.000380) | 0.430806 (0.000342) | 0.430757 (0.000233) | -0.005 | -0.064 |
| 10% | 0.435615 (0.000272) | 0.434893 (0.000120) | 0.435025 (0.000400) | +0.013 | -0.059 |
| 15% | 0.437996 (0.000448) | 0.437464 (0.000261) | 0.437717 (0.000170) | +0.025 | -0.028 |
| 20% | 0.439715 (0.000293) | 0.439364 (0.000280) | 0.439310 (0.000251) | -0.005 | -0.041 |

Decision: no ratio makes legacy Graph consistently superior. 15% has the largest positive Graph-Adam mean delta, but remains below Random; do not freeze final ratios until corrected score caches are rebuilt.
