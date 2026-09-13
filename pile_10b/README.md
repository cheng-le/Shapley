# Pile 10B token 单机四卡部署

## 本入口实际运行什么

目标硬件是 **4 张各 48GB GPU**，不是 192GB 系统内存。脚本独立检查主机 RAM、可用磁盘、CUDA/BF16，并锁定数据与模型的远端 commit。默认 GPT-2、10,000,000,000 个训练输入 token、1024 长度、三个 seed、只训练最终 LayerNorm，输出同一训练轨迹上的 Adam-aware 一阶及无 HVP Graph 分数。

**这是新建的大规模归因入口，不是全部论文实验的一键复现。** 默认不运行二阶 HVP、其他主表基线、子集重训和人工噪声实验；报告中的 validation loss 来自共享的全数据训练模型，不能解释为 Graph 优于 Adam-aware 的下游效果。若目标是全参数预训练，需要另行实现与验证全参数归因；不能把 `ln_f` 的结果冒充全参数结果。

默认沿用项目数据源 `monology/pile-uncopyrighted`。它是 Pile 的一个版本，**10B token 也不是整个 Pile**。源说明见 [数据集卡](https://huggingface.co/datasets/monology/pile-uncopyrighted/blob/main/README.md)。可通过配置的 `local_jsonl` 提供自己的完整 Pile 文件列表，支持 `.jsonl`、`.jsonl.gz`、`.jsonl.zst`，每行必须有字符串 `text`，可带 `meta.pile_set_name`。不会在数据源不可用时偷偷换成其他语料。

## 新服务器一键运行

先准备 Linux、Python 3.10–3.12、可用 NVIDIA 驱动、Git、C++ 编译器、Python venv、`flock`，并保证能访问 Hugging Face 与 Python 包仓库。`hnswlib` 需要本地编译。以下假设仓库已上传：

```bash
git clone <你的GitHub仓库URL> Shapley
cd Shapley
WORK_DIR=/大容量磁盘/pile10b bash pile_10b/run.sh
```

默认创建独立 `.venv-pile10b` 并安装固定依赖（PyTorch 2.6.0 CUDA 12.4 wheel）；不会修改系统 Python。如驱动不兼容，请在驱动就绪后再跑，或自行准备并验证兼容环境：

```bash
SKIP_INSTALL=1 PYTHON=/path/to/env/bin/python \
WORK_DIR=/大容量磁盘/pile10b bash pile_10b/run.sh
```

私有/受限数据源的访问凭据通过 `HF_TOKEN` 环境变量或 Hugging Face 本机登录配置提供，**不要写入配置、Git URL 或仓库**。`HF_HOME` 可指向大容量缓存盘。默认使用四张可见 GPU；可显式设置 `CUDA_VISIBLE_DEVICES` 和 `GPUS`，续训时 GPU 数量必须不变。

长任务建议在 tmux 中执行同一命令。日志、环境版本、输入配置和完成状态都保存在 `WORK_DIR`。脚本有进程锁，拒绝两个入口同时写一个目录。

## 先检查，再接受长任务

```bash
PREFLIGHT_ONLY=1 WORK_DIR=/大容量磁盘/pile10b bash pile_10b/run.sh
```

默认预算约产生 976.6 万个 token 块。仅训练 token 的 uint32 文件约 40GB；抽取数据库另存 token，特征、图、索引及检查点还有额外开销。脚本按配置计算并检查 RAM/磁盘需求，**建议至少预留 200GB 工作磁盘**，Hugging Face 缓存和本地原始数据另计。系统 RAM 不是显存，必须看 `hardware.json` 的实际检查结果。估算不等于峰值资源保证。

流程依次为抽取、四卡特征、CPU 全局 HNSW、四卡同步训练、汇总。训练会先测量少量真实 step，写入 `seed_*/throughput.json`；每 seed 预计剩余时间超过默认 168 小时时，保存检查点并停止。确认成本后原命令前加 `ALLOW_LONG_RUN=1` 续跑。这个测量发生在特征/建图完成后，不涵盖它们的耗时，也不能保证 10B 很快跑完。

三个 seed 是 **每个 seed 各遍历 10B 输入 token**，累计训练遍历 30B；特征前向、归因查询和逐样本归因还有额外计算。四卡合作训练一个 seed，然后依次运行后续 seed，不是把 10B 拆成四个独立模型。

## 数据与方法口径

- 以 GPT-2 tokenizer 的实际 token ID 精确计数，包括文档 EOS，不计 padding；最后一个块允许不足 1024。报告同时给出可预测 token 数，因为每块首 token 不作为预测目标。
- 文本 SHA256 决定 train/query/val 三个互斥分区，SQLite 去除精确重复；最终验证文本不用于归因查询。不做跨语料近重复检测。每个分区按源顺序提取到预算，因此不是对整个 Pile 的均匀随机抽样。
- token 打包可以跨文档边界，归因单位是 **token 块，不是原始文档**。`documents.sqlite` 保留原文 SHA256、源行序号、domain 和 token 数，可按分区源序重建块的来源区间。
- 图表示来自初始 GPT-2 的平均 hidden state，经固定 128 维高斯投影；HNSW 在全部训练块上建全局近似图，不是各卡局部图。存储采用磁盘 memmap，不构造全局 N×N 相似度矩阵。
- `ann_audit.json` 报告投影空间内、固定随机查询点上的 ANN recall@k；低于阈值会停止。它不是投影相对原始 768 维表示的保真度测量。多线程 HNSW 重建不保证位级一致，因此保存最终边和哈希。
- 复用在线 Graph 的 history/pair 与尺度匹配公式；全局批次一起修正分数，各卡同步最终 LayerNorm 梯度与参数。默认全局 batch=4×4=16。与旧 1k 的 batch=8、128 长度不同，不能混表当作同一协议。
- 训练顺序为固定种子的可逆仿射置换，覆盖每个块且不分配巨大随机排列；不是所有排列中的均匀采样。默认 AdamW 参数沿用在线实验；它不保证在 10B 规模上达到最优训练效果。
- 无 HVP 路径保留不利结果，不自动搜索让 Graph 获胜的配置。不生成没有标签支持的 noise AUROC、domain recall 或不存在的基线数据。

## 断点恢复与产物

重复相同命令即可续跑。配置变化会被拒绝，应更换 `WORK_DIR`。抽取阶段每 1000 源行事务提交；中断后重放并跳过已提交源行，远端流可能需要重新传输前缀，**不是服务器端 HTTP 游标续传**。完成的数据缓存会复用。

特征按块写入并标记完成；训练检查点原子保存模型、优化器、每 rank RNG、历史分数及下一步位置。最多重算一个检查点区间。HNSW **未完成的建图需要重建**，完成后复用保存的邻居与权重。请不要手动修改缓存或检查点。

主要产物：`data.json`、`documents.sqlite`、`provenance.json`、`hardware.json`、`environment.txt`、`ann_audit.json`、`seed_*/checkpoint.pt`、`seed_*/adam1_scores.npy`、`seed_*/graph_scores.npy`、`seed_*/complete.json`、`report.json`。单 seed 的样本标准差记为 null，不伪造为 0。

## 上传当前项目代码到 GitHub

当前工作区包含多 GB 数据和缓存。使用专门导出工具保留现有 Python、Shell、TeX、Markdown 源码及部署配置；不携带数据、模型、结果二进制、PDF、缓存或本机环境。这个源码包能部署新入口，但旧实验图表需要另行归档，不等于完整实验数据备份。

```bash
python3 -m pile_10b.export_source --output /tmp/Shapley-github
cd /tmp/Shapley-github
git init -b main
git add .
git diff --cached --stat
git commit -m "Add reproducible Pile 10B deployment pipeline"
git remote add origin <你的空GitHub仓库URL>
git push -u origin main
```

请先在 GitHub 建立空仓库；论文尚未公开时建议设为 private。导出工具会检查常见 token/私钥模式并生成 `SOURCE_MANIFEST.json`，但它不能替代发布前人工检查。工具不读取或上传 Hugging Face/GitHub 登录凭据，也不会替你创建远端仓库或自动推送。

## 验证

```bash
python -m unittest pile_10b.test_pipeline -v
python -m pile_10b.integration_test
```

集成测试需要本机已有 GPT-2 tokenizer 缓存，使用随机初始化的微型模型、本地合成文本和双进程 CPU/Gloo，验证中断续跑与连续运行一致，**不是科学实验结果，也不替代四卡 CUDA/NCCL 测试**。正式运行不允许自动退回 CPU。生产分布式使用 [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)，近邻使用 [HNSWlib](https://github.com/nmslib/hnswlib)。
