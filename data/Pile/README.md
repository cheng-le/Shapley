# Pile Mini Subset Workspace

这个目录用于准备 Pile+GPT2 实验所需的小规模 Pile 子集。

当前状态：没有成功下载真实 Pile 原始分片。历史 `the-eye.eu` 地址在本服务器上返回 404/SSL 错误，因此这里先建立 22 子集抽样工作区和 manifest。

目录：

- `raw/`: 放你手动下载到的 Pile `.jsonl` 或 `.jsonl.zst` 分片。
- `subsets/`: 脚本会为 22 个 Pile 子集分别输出抽样文件。
- `train.jsonl`: GPT2 代码默认读取的训练子集。
- `val.jsonl`: GPT2 代码默认读取的验证子集。
- `pile_22_subsets_manifest.json`: 22 个子集清单和当前下载状态。

生成方式：

```bash
/home/lcheng/anaconda3/envs/torch_5090/bin/python pile_gpt2/prepare_pile_22_subset.py \
  --raw-dir data/Pile/raw \
  --out-dir data/Pile \
  --per-subset-train 50 \
  --per-subset-val 10
```

脚本要求输入文件每行是 Pile 风格 JSON，例如：

```json
{"text": "...", "meta": {"pile_set_name": "Wikipedia (en)"}}
```

如果是单子集文件，也可以只包含 `text`，再用文件名或 `--default-subset` 指定来源。
