# CIFAR10 Table 4 comparison

This directory reproduces the mislabeled-data detection protocol from Table 4
of *Data Shapley in One Training Run* and adds the Adam-aware and graph-aware
In-Run variants developed in this repository.

## Protocol

- Uniformly sample 1,000 CIFAR10 training examples per run.
- Replace exactly 10% of labels with a different uniformly sampled class.
- Repeat for seeds 0, 1, and 2.
- Generate data-attribution scores using the same models, optimizer, schedule,
  and method settings as the completed Table 5 experiment.
- Treat mislabeled examples as the positive class and use negative attribution
  value as the anomaly score. Report mean AUROC and population standard
  deviation over the three corrupted datasets.

The model is ImageNet-pretrained ResNet18 trained for 50 epochs with Adam,
learning rate 0.001, weight decay 0.0001, label smoothing 0.1, batch size 64,
and learning-rate decay by 0.1 every 10 epochs. The score-cache fingerprint is
locked to `16a6f42d74e8` so Table 4 cannot silently drift from Table 5.

## Run

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python table4_cifar_mislabel_detection/run_table4.py
```

Existing Table 5 score vectors are reused because Table 4 changes only the
evaluation metric; absent caches are computed by their owning method module.
Outputs are written to `results/table4_results.{json,csv,md,txt}`. The Markdown
and plain-text outputs use the same compact `mean (standard deviation)` layout
as the completed Table 5 result.

The exact-versus-approximate implementation boundaries are unchanged from
`table5_cifar_comparison/README.md`: KNN-Shapley uses the exact recurrence,
while TRAK, LESS, influence functions, and In-Run methods use the repository's
documented last-layer approximations.
