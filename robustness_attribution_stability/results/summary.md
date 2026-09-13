# Robustness and Attribution Stability Results

## Corruption Robustness

| Method | Pile AUROC | Pile Top-10% contamination | CIFAR-10 AUROC | CIFAR-10 Top-10% contamination |
|---|---:|---:|---:|---:|
| Adam-aware 1st Order In-Run | 0.559 (0.002) | 0.093 (0.002) | 0.500 (0.019) | 0.130 (0.024) |
| Graph Adam-aware 1st Order In-Run | 0.563 (0.002) | 0.092 (0.002) | 0.627 (0.015) | 0.070 (0.008) |
| Adam-aware 2nd Order In-Run | 0.559 (0.002) | 0.093 (0.002) | 0.491 (0.009) | 0.127 (0.009) |
| Graph Adam-aware 2nd Order In-Run | 0.562 (0.001) | 0.093 (0.002) | 0.623 (0.007) | 0.067 (0.012) |

## Pile Cross-Seed Stability

| Method | Spearman | Kendall tau | Top-10% overlap | Standardized dispersion |
|---|---:|---:|---:|---:|
| Adam-aware 1st Order In-Run | 0.164 | 0.110 | 0.164 | 0.623 |
| Graph Adam-aware 1st Order In-Run | 0.183 | 0.123 | 0.175 | 0.615 |
| Adam-aware 2nd Order In-Run | 0.165 | 0.111 | 0.166 | 0.622 |
| Graph Adam-aware 2nd Order In-Run | 0.185 | 0.124 | 0.177 | 0.614 |

## Corrupted-Node Graph Change

| Dataset | Order | Standardized score delta | Promotion rate |
|---|---:|---:|---:|
| Pile | 1st | -0.010 | 0.406 |
| Pile | 2nd | -0.010 | 0.408 |
| CIFAR-10 | 1st | -0.416 | 0.013 |
| CIFAR-10 | 2nd | -0.401 | 0.010 |

Lower contamination, dispersion, corrupted-node delta, and promotion rate are better; higher AUROC and rank correlations are better.
Pile Graph v2 is analyzed as the frozen approximation documented by the main experiment, not as a strict online implementation.
