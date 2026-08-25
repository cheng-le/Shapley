# Pile + GPT2 In-Run Data Shapley

这个目录用于实现论文主实验风格的语言模型版本：文本数据来自 Pile 或本地文本文件，模型为 HuggingFace GPT2，分数计算沿用项目中的 SGD、Adam-aware 和 Graph 版本。

## 依赖

当前环境需要额外安装：

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python -m pip install transformers datasets accelerate
```

如果不能联网下载 Pile 或 GPT2，可以先用本地文本文件调试：每行一个文本样本。


## 关于 EleutherAI/pile 和 datasets 5.0

`EleutherAI/pile` 的 HuggingFace 入口依赖旧 Python loading script，并且脚本会从 `the-eye.eu` 拉取原始 Pile 大分片。当前常见问题有两个：

```text
Dataset scripts are no longer supported, but found pile.py
CERTIFICATE_VERIFY_FAILED ... the-eye.eu ... 00.jsonl.zst
```

因此代码默认不会自动触发 `EleutherAI/pile` 远程全量下载。若你确认要走旧脚本远程下载，需要显式加 `--allow-remote-pile-download`。

解决方式二选一：

1. 降级 datasets 到仍支持 loading script 的版本：

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python -m pip install 'datasets<4'
```

2. 推荐：手动下载/整理 Pile 的 jsonl/parquet 小分片，然后用本地文件运行：

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python pile_gpt2/first_order/in_run_sgd_first_order.py \
  --local-train-text data/Pile/train.jsonl \
  --local-val-text data/Pile/val.jsonl \
  --dataset-format jsonl --text-field text
```

## 快速检查

推荐先从 `monology/pile-uncopyrighted` 流式抽一个本地小子集，不要直接下载 335GB 全量数据：

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python pile_gpt2/download_pile_uncopyrighted_subset.py   --per-subset-train 50   --per-subset-val 10   --out-dir data/Pile
```

生成后会得到：

```text
data/Pile/train.jsonl
data/Pile/val.jsonl
data/Pile/subsets/*.jsonl
```

注意：`pile-uncopyrighted` 移除了 `Books3`、`BookCorpus2`、`OpenSubtitles`、`YoutubeSubtitles` 和 `OpenWebText2`，所以它不能完全等同原始 22 子集 Pile，但适合当前 GPT2 Shapley 实验。

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python pile_gpt2/first_order/in_run_adam_first_order.py \
  --local-train-text data/Pile/train.txt \
  --local-val-text data/Pile/val.txt \
  --train-limit 16 --val-limit 16 --batch-size 1 --block-size 32 \
  --val-samples 2 --mc-batch-size 2 --mc-permutations 2 --prune-epochs 1
```

## 论文对齐提示

严格对齐论文主实验时，需要使用 Pile 子集、GPT2/Pythia 模型和论文中的训练预算。完整 GPT2 的 per-sample gradient 和 HVP 非常重，建议先用小 `train-limit` 和短 `block-size` 验证流程，再逐步扩大规模。

Graph 方法当前用 token 哈希直方图构造文本 KNN 图，是为了把你的图稳定化思想迁移到文本场景；后续可以替换为 GPT2 hidden-state embedding 图。
