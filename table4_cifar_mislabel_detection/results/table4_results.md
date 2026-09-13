# CIFAR10 Table 4 Results

```text
Method                             AUROC        
Retraining-based Data Shapley      0.500 (0.042)
Empirical Influence Function       0.507 (0.008)
Datamodels                         0.504 (0.041)
KNN-Shapley                        0.563 (0.054)
Influence function                 0.447 (0.040)
LESS                               0.523 (0.027)
1st Order In-Run Data Shapley      0.512 (0.022)
2nd Order In-Run Data Shapley      0.500 (0.035)
TRAK (1 cpt)                       0.488 (0.030)
TRAK (5 cpts)                      0.478 (0.028)
TRAK (15 cpts)                     0.477 (0.031)
TRAK (25 cpts)                     0.477 (0.031)
Adam-aware 1st Order In-Run        0.500 (0.019)
Adam-aware 2nd Order In-Run        0.491 (0.009)
Graph Adam-aware 1st Order In-Run  0.627 (0.015)
Graph Adam-aware 2nd Order In-Run  0.623 (0.007)
```

## Paper Reference Comparison

| Method | Reproduced | Paper | Delta |
|---|---:|---:|---:|
| Retraining-based Data Shapley | 0.500 (0.042) | 0.582 (0.029) | -0.082 |
| Empirical Influence Function | 0.507 (0.008) | 0.552 (0.017) | -0.045 |
| Datamodels | 0.504 (0.041) | 0.520 (0.008) | -0.016 |
| KNN-Shapley | 0.563 (0.054) | 0.760 (0.018) | -0.197 |
| Influence function | 0.447 (0.040) | 0.654 (0.054) | -0.207 |
| LESS | 0.523 (0.027) | 0.550 (0.032) | -0.027 |
| 1st Order In-Run Data Shapley | 0.512 (0.022) | 0.678 (0.045) | -0.166 |
| 2nd Order In-Run Data Shapley | 0.500 (0.035) | 0.680 (0.048) | -0.180 |
| TRAK (1 cpt) | 0.488 (0.030) | 0.511 (0.012) | -0.023 |
| TRAK (5 cpts) | 0.478 (0.028) | 0.542 (0.025) | -0.064 |
| TRAK (15 cpts) | 0.477 (0.031) | 0.609 (0.028) | -0.132 |
| TRAK (25 cpts) | 0.477 (0.031) | 0.617 (0.021) | -0.140 |

Standard deviations use the population convention over three runs.
