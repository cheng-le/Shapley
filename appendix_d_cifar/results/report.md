# Appendix D: CIFAR Sensitivity

Frozen ImageNet ResNet18 features, trainable linear head only. 3 data seeds crossed with 3 training seeds. 43 fixed graph configurations plus a random reference per trajectory. Each selection retains 200/1000 examples and is genuinely retrained for 50 head-training epochs. Identical selected sets within a seed pair reuse one fit and are explicitly marked; they are not independent training runs.

Full denotes the optional initial-gradient-gated extension; no_gate is the semantic-only paper weighting. Gate gradients are computed at initialization, not taken from future trajectory states. Scores replay the exact recorded five-epoch first-order Adam trajectory; no HVP or second-order cache is used.

Validation is 1000 held-out CIFAR TRAIN images (128 used by attribution); test is 2000 official TEST images. There is no validation/test overlap. No hyperparameter was selected on the final test outcomes; this is a descriptive sweep, not a tuned main result.

Mean (sample SD) across the 9 crossed data/training cells; these are NOT 9 independent dataset samples. All outcomes, including null and adverse ones, are retained.

| Variant | Test accuracy | Test loss | Noise AUROC | Selected noise | Coverage | ANN recall |
|---|---:|---:|---:|---:|---:|---:|
| k5_a0_l0 | 0.14328 (0.02970) | 2.37923 (0.01328) | 0.57095 (0.00927) | 0.07333 (0.02000) | 0.83733 (0.02446) | 1.00000 (0.00000) |
| k5_a0_l0.05 | 0.15178 (0.03575) | 2.37894 (0.01213) | 0.57095 (0.00927) | 0.07444 (0.01845) | 0.84067 (0.02467) | 1.00000 (0.00000) |
| k5_a0_l0.1 | 0.15717 (0.03018) | 2.37677 (0.01425) | 0.57095 (0.00927) | 0.07500 (0.01803) | 0.84433 (0.02230) | 1.00000 (0.00000) |
| k5_a0.2_l0 | 0.14972 (0.03429) | 2.37556 (0.00747) | 0.63908 (0.00897) | 0.05500 (0.01500) | 0.82278 (0.01328) | 1.00000 (0.00000) |
| k5_a0.2_l0.05 | 0.15133 (0.03322) | 2.37485 (0.00831) | 0.63908 (0.00897) | 0.05500 (0.01500) | 0.82467 (0.01405) | 1.00000 (0.00000) |
| k5_a0.2_l0.1 | 0.14483 (0.03371) | 2.37287 (0.00767) | 0.63908 (0.00897) | 0.05556 (0.01530) | 0.82856 (0.01202) | 1.00000 (0.00000) |
| k5_a0.4_l0 | 0.16572 (0.02119) | 2.37118 (0.00540) | 0.69525 (0.00967) | 0.02833 (0.01000) | 0.80178 (0.01929) | 1.00000 (0.00000) |
| k5_a0.4_l0.05 | 0.16333 (0.02061) | 2.36932 (0.00670) | 0.69525 (0.00967) | 0.02833 (0.01000) | 0.80567 (0.02056) | 1.00000 (0.00000) |
| k5_a0.4_l0.1 | 0.15417 (0.02925) | 2.36807 (0.00573) | 0.69525 (0.00967) | 0.02944 (0.00982) | 0.80744 (0.02000) | 1.00000 (0.00000) |
| k10_a0_l0 | 0.14328 (0.02970) | 2.37923 (0.01328) | 0.57095 (0.00927) | 0.07333 (0.02000) | 0.83733 (0.02446) | 1.00000 (0.00000) |
| k10_a0_l0.05 | 0.15167 (0.03550) | 2.37863 (0.01305) | 0.57095 (0.00927) | 0.07333 (0.01871) | 0.84022 (0.02477) | 1.00000 (0.00000) |
| k10_a0_l0.1 | 0.15417 (0.03357) | 2.37860 (0.01226) | 0.57095 (0.00927) | 0.07500 (0.01854) | 0.84478 (0.02129) | 1.00000 (0.00000) |
| k10_a0.2_l0 | 0.15933 (0.03253) | 2.37603 (0.00782) | 0.64680 (0.00864) | 0.05000 (0.01118) | 0.81011 (0.01681) | 1.00000 (0.00000) |
| full | 0.15122 (0.03310) | 2.37609 (0.00776) | 0.64680 (0.00864) | 0.05167 (0.01199) | 0.81322 (0.01671) | 1.00000 (0.00000) |
| k10_a0.2_l0.1 | 0.14828 (0.03486) | 2.37536 (0.00767) | 0.64680 (0.00864) | 0.05278 (0.01277) | 0.81733 (0.01555) | 1.00000 (0.00000) |
| k10_a0.4_l0 | 0.16578 (0.02435) | 2.36935 (0.00326) | 0.70761 (0.00971) | 0.02389 (0.00782) | 0.79556 (0.02489) | 1.00000 (0.00000) |
| k10_a0.4_l0.05 | 0.16739 (0.02622) | 2.36862 (0.00318) | 0.70761 (0.00971) | 0.02333 (0.00707) | 0.79767 (0.02630) | 1.00000 (0.00000) |
| k10_a0.4_l0.1 | 0.15783 (0.02682) | 2.36667 (0.00296) | 0.70761 (0.00971) | 0.02500 (0.00612) | 0.80000 (0.02423) | 1.00000 (0.00000) |
| k20_a0_l0 | 0.14328 (0.02970) | 2.37923 (0.01328) | 0.57095 (0.00927) | 0.07333 (0.02000) | 0.83733 (0.02446) | 1.00000 (0.00000) |
| k20_a0_l0.05 | 0.15228 (0.03546) | 2.37913 (0.01228) | 0.57095 (0.00927) | 0.07389 (0.01867) | 0.84067 (0.02460) | 1.00000 (0.00000) |
| k20_a0_l0.1 | 0.15333 (0.03276) | 2.37919 (0.01224) | 0.57095 (0.00927) | 0.07500 (0.01854) | 0.84389 (0.02139) | 1.00000 (0.00000) |
| k20_a0.2_l0 | 0.15539 (0.02789) | 2.37354 (0.00827) | 0.65184 (0.00873) | 0.05000 (0.01031) | 0.80911 (0.01772) | 1.00000 (0.00000) |
| k20_a0.2_l0.05 | 0.15306 (0.02409) | 2.37334 (0.00687) | 0.65184 (0.00873) | 0.05056 (0.01074) | 0.81156 (0.01870) | 1.00000 (0.00000) |
| k20_a0.2_l0.1 | 0.14894 (0.02887) | 2.37367 (0.00784) | 0.65184 (0.00873) | 0.05056 (0.01074) | 0.81456 (0.01766) | 1.00000 (0.00000) |
| k20_a0.4_l0 | 0.17422 (0.02463) | 2.36822 (0.00487) | 0.71556 (0.00977) | 0.02111 (0.00697) | 0.79144 (0.02514) | 1.00000 (0.00000) |
| k20_a0.4_l0.05 | 0.16750 (0.03042) | 2.36683 (0.00522) | 0.71556 (0.00977) | 0.02167 (0.00707) | 0.79511 (0.02294) | 1.00000 (0.00000) |
| k20_a0.4_l0.1 | 0.17056 (0.02553) | 2.36597 (0.00518) | 0.71556 (0.00977) | 0.02167 (0.00707) | 0.79900 (0.02636) | 1.00000 (0.00000) |
| no_gate | 0.14956 (0.03285) | 2.38494 (0.00962) | 0.55829 (0.01373) | 0.07111 (0.01816) | 0.81211 (0.01918) | 1.00000 (0.00000) |
| no_history | 0.15011 (0.03440) | 2.37817 (0.01436) | 0.58381 (0.00888) | 0.07000 (0.01768) | 0.83567 (0.02087) | 1.00000 (0.00000) |
| no_pair | 0.14989 (0.03087) | 2.37672 (0.00873) | 0.63580 (0.00877) | 0.05500 (0.01369) | 0.81822 (0.01133) | 1.00000 (0.00000) |
| no_scale | 0.17333 (0.03128) | 2.33885 (0.00462) | 0.82120 (0.01006) | 0.01056 (0.00527) | 0.81967 (0.03134) | 1.00000 (0.00000) |
| resnet32 | 0.14717 (0.03669) | 2.37477 (0.00856) | 0.61804 (0.00820) | 0.05889 (0.01387) | 0.83344 (0.02109) | 1.00000 (0.00000) |
| pixels8 | 0.14061 (0.02905) | 2.37879 (0.01007) | 0.59829 (0.00761) | 0.06556 (0.01379) | 0.83544 (0.01675) | 1.00000 (0.00000) |
| sym_mean | 0.15444 (0.03440) | 2.37285 (0.00658) | 0.63897 (0.00919) | 0.05389 (0.01318) | 0.83144 (0.01676) | 1.00000 (0.00000) |
| sym_mutual | 0.14356 (0.03496) | 2.37381 (0.00650) | 0.62488 (0.00882) | 0.05722 (0.01583) | 0.83544 (0.02013) | 1.00000 (0.00000) |
| ann2 | 0.15022 (0.03420) | 2.37354 (0.00669) | 0.63533 (0.00774) | 0.05444 (0.01446) | 0.82122 (0.01773) | 0.53417 (0.00877) |
| ann8 | 0.14422 (0.02845) | 2.37553 (0.00884) | 0.64448 (0.00823) | 0.05389 (0.01341) | 0.82000 (0.00987) | 0.81153 (0.01289) |
| ann32 | 0.15467 (0.03199) | 2.37609 (0.00757) | 0.64701 (0.00846) | 0.05167 (0.01199) | 0.81489 (0.01721) | 0.94523 (0.00165) |
| ann128 | 0.15717 (0.03244) | 2.37675 (0.00843) | 0.64679 (0.00855) | 0.05167 (0.01199) | 0.81344 (0.01651) | 0.98893 (0.00101) |
| ann8_seed1 | 0.14761 (0.03227) | 2.37755 (0.00782) | 0.64410 (0.00853) | 0.05222 (0.01253) | 0.81600 (0.01444) | 0.82023 (0.00211) |
| exact_seed1 | 0.15122 (0.03310) | 2.37609 (0.00776) | 0.64680 (0.00864) | 0.05167 (0.01199) | 0.81322 (0.01671) | 1.00000 (0.00000) |
| ann8_seed2 | 0.15494 (0.03350) | 2.37643 (0.00787) | 0.64253 (0.00834) | 0.05333 (0.01250) | 0.81911 (0.01238) | 0.80593 (0.00200) |
| exact_seed2 | 0.15122 (0.03310) | 2.37609 (0.00776) | 0.64680 (0.00864) | 0.05167 (0.01199) | 0.81322 (0.01671) | 1.00000 (0.00000) |
| random | 0.22006 (0.08508) | 2.22128 (0.00477) | 0.50920 (0.02811) | 0.09722 (0.01439) | 0.95544 (0.00794) | 1.00000 (0.00000) |

## Interpretation boundaries

- This is not Table 5 full-network training. Alternative graph embeddings change only the graph; the classification representation remains resnet128.
- Directed/mean/mutual refer to propagation adjacency. Selection always uses the symmetric redundancy objective. Diversity is evaluated on the SAME reference graph for every configuration.
- HNSW uses M=4, efConstruction=20, one thread. Graph seed controls insertion permutation; training/data seeds stay fixed. Exact graph seeds are a deterministic negative control, checked for bitwise-identical scores.
- At very low efSearch, HNSW may return fewer than k valid neighbors. Missing neighbors carry zero weight and are NOT filled from the exact graph. Effective degree is recorded alongside recall.
- Seed-factor results comprise a fully crossed 3 x 3 x 3 ANN design. Marginal comparisons are descriptive; data-seed subsets can overlap and pairwise rank correlations are dependent.
- Existing Pile results remain available under efficiency_scalability_ablations and are not replaced by this modality change. The Pile 1k online study had near-chance noise AUROC and all selected-subset losses worse than its pretrained reference.
- Retention is fixed at 20%. Larger/full-network or other-budget conclusions require further experiments. No artificial variance is added to deterministic controls.
