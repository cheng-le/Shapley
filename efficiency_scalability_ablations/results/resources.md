# Resource Measurements

Times exclude cache loading, embedding/gradient extraction and model training. Pipeline time sums independently timed graph build, gate, replay and selection; pointwise sums score replay and Top-k. These are not full attribution times.

| N | Method | Cached pipeline seconds | Replay ms/batch | CUDA peak MiB | Process RSS peak MiB |
|---:|---|---:|---:|---:|---:|
| 1000 | pointwise | 0.000 (0.000) | 0.003 (0.000) | 0.000 (0.000) | 853.294 (0.010) |
| 1000 | exact1 | 0.090 (0.005) | 0.055 (0.000) | 7.784 (0.000) | 939.078 (9.637) |
| 1000 | exact2 | 0.100 (0.004) | 0.140 (0.002) | 7.784 (0.000) | 939.078 (9.637) |
| 1000 | ann1 | 0.153 (0.005) | 0.055 (0.001) | 0.000 (0.000) | 942.355 (13.637) |
| 1000 | ann2 | 0.163 (0.005) | 0.139 (0.001) | 0.000 (0.000) | 942.355 (13.637) |
| 10000 | pointwise | 0.003 (0.000) | 0.001 (0.000) | 0.000 (0.000) | 893.173 (0.253) |
| 10000 | exact1 | 0.733 (0.006) | 0.053 (0.001) | 69.594 (0.000) | 1030.688 (16.209) |
| 10000 | exact2 | 0.833 (0.007) | 0.134 (0.000) | 69.594 (0.000) | 1030.688 (16.209) |
| 10000 | ann1 | 1.669 (0.020) | 0.053 (0.001) | 0.000 (0.000) | 1051.507 (2.407) |
| 10000 | ann2 | 1.774 (0.022) | 0.135 (0.001) | 0.000 (0.000) | 1051.507 (2.407) |
| 50000 | pointwise | 0.012 (0.000) | 0.001 (0.000) | 0.000 (0.000) | 1070.066 (0.153) |
| 50000 | exact1 | 3.950 (0.046) | 0.055 (0.004) | 342.203 (0.000) | 1688.613 (14.787) |
| 50000 | exact2 | 4.454 (0.034) | 0.137 (0.002) | 342.203 (0.000) | 1688.613 (14.787) |
| 50000 | ann1 | 12.063 (0.042) | 0.054 (0.001) | 0.000 (0.000) | 1695.361 (17.428) |
| 50000 | ann2 | 12.591 (0.033) | 0.140 (0.004) | 0.000 (0.000) | 1695.361 (17.428) |
| 100000 | pointwise | 0.023 (0.000) | 0.001 (0.000) | 0.000 (0.000) | 1290.900 (0.073) |
| 100000 | exact1 | 8.416 (0.032) | 0.053 (0.001) | 684.344 (0.000) | 2522.901 (28.307) |
| 100000 | exact2 | 9.466 (0.036) | 0.137 (0.001) | 684.344 (0.000) | 2522.901 (28.307) |
| 100000 | ann1 | 27.729 (0.424) | 0.054 (0.001) | 0.000 (0.000) | 2534.423 (30.304) |
| 100000 | ann2 | 28.772 (0.413) | 0.137 (0.003) | 0.000 (0.000) | 2534.423 (30.304) |

CUDA peak is the maximum incremental PyTorch allocation across stages, not total device use. RSS is absolute process peak and includes frameworks and resident input/graph arrays; stages share a worker, so it is not standalone per-method RAM. Replay-batch cost is total replay divided by ceil(N/8), not measured online latency.

## ANN Fidelity and Storage

| N | Recall@10 | Score Spearman order 1 / 2 | Selection overlap order 1 / 2 | Input tensors MiB | Sparse graph MiB | Graph+scores MiB | HNSW index MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.99910 (0.00000) | 0.99986 (0.00000) / 0.99985 (0.00000) | 1.00000 (0.00000) / 1.00000 (0.00000) | 4.402 | 0.076 | 0.126 | 3.189 |
| 10000 | 0.99792 (0.00001) | 0.99989 (0.00000) / 0.99988 (0.00000) | 0.99900 (0.00000) / 0.99900 (0.00000) | 44.022 | 0.763 | 1.259 | 31.889 |
| 50000 | 0.99716 (0.00005) | 0.99989 (0.00001) / 0.99989 (0.00001) | 0.99947 (0.00012) / 0.99927 (0.00012) | 220.108 | 3.815 | 6.294 | 159.466 |
| 100000 | 0.99632 (0.00001) | 0.99980 (0.00000) / 0.99980 (0.00000) | 0.99873 (0.00006) / 0.99817 (0.00006) | 440.216 | 7.629 | 12.589 | 318.929 |

Storage columns are byte counts of array payloads and the actual serialized HNSW index, not compressed NPZ file sizes. Original on-disk input cache sizes are recorded in provenance.json. Near-tied cosine similarities can change exact neighbor identities. ANN is not tuned against diagnostics.

## Validation

All 12 timing cells (168 stages), 168 ablation cells and 84 clean score vectors were checked for completeness. Replay parity and zero-strength limits passed. Source hashes and input-cache sizes/timestamps remained unchanged during the run.
