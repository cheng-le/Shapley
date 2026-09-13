"""Audit and report fixed-cache stability/noise diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(values, digits=5):
    return "%.*f (%.*f)" % (digits, np.mean(values), digits, np.std(values, ddof=1))


def main():
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest() == provenance["script_sha256"]
    assert hashlib.sha256((ROOT / "appendix_d_cifar/run.py").read_bytes()).hexdigest() == provenance["graph_code_sha256"]
    for path, digest in provenance["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    nodes, edges = read("nodes.csv"), read("edges.csv")
    contamination, stability, effects = [read(name + ".csv") for name in ("contamination", "stability_summary", "propagation_effects")]
    assert len(nodes) == 9000 and len(edges) == 90000 and len(contamination) == 486
    assert len(read("score_distribution.csv")) == 63000
    for ds in range(3):
        for ts in range(3):
            rows = [r for r in nodes if int(r["data_seed"]) == ds and int(r["train_seed"]) == ts]
            assert len(rows) == 1000 and sum(int(r["corrupted"]) for r in rows) == 100
            by_id = {int(r["local_id"]): r for r in rows}
            for row in [r for r in edges if int(r["data_seed"]) == ds and int(r["train_seed"]) == ts]:
                assert row["source_corrupted"] == by_id[int(row["source"])]["corrupted"]
                assert row["recipient_corrupted"] == by_id[int(row["recipient"])]["corrupted"]
                assert float(row["gated_effective_weight"]) >= 0
            graph_values = np.array([float(by_id[i]["graph_score"]) for i in range(1000)])
            for row in [r for r in contamination if int(r["data_seed"]) == ds and int(r["train_seed"]) == ts and r["method"] == "full"]:
                chosen = np.argsort(-graph_values, kind="stable")[:int(row["count"])]
                assert sum(int(by_id[int(i)]["corrupted"]) for i in chosen) == int(row["corrupted_selected"])
    bucket_rows = read("buckets.csv")
    for ds in range(3):
        for ts in range(3):
            for variable in ("label_agreement", "gradient_agreement", "degree", "density"):
                group = [r for r in bucket_rows if int(r["data_seed"]) == ds and int(r["train_seed"]) == ts and r["variable"] == variable]
                assert sum(int(r["count"]) for r in group) == 1000
                assert sum(int(r["corrupted_count"]) for r in group) == 100
    flow = []
    for ds in range(3):
        for ts in range(3):
            group = [r for r in edges if int(r["data_seed"]) == ds and int(r["train_seed"]) == ts]
            for recipient in (0, 1):
                to = [r for r in group if int(r["recipient_corrupted"]) == recipient]
                for source in (0, 1):
                    part = [r for r in to if int(r["source_corrupted"]) == source]
                    for key in ("semantic_weight", "gated_effective_weight"):
                        mass = sum(float(r[key]) for r in part)
                        total = sum(float(r[key]) for r in to)
                        flow.append(dict(data_seed=ds, train_seed=ts, recipient_corrupted=recipient,
                            source_corrupted=source, weighting=key, mass=mass,
                            mean_mass_per_recipient=mass / (100 if recipient else 900),
                            share_within_recipient_type=mass / max(total, 1e-12)))
    write("noise_flow.csv", flow)
    names = {"k10_a0_l0": "Adam-aware 一阶", "full": "Graph（带门控）", "no_gate": "无 gate",
             "no_history": "无历史", "no_pair": "无 pair", "no_scale": "无尺度匹配", "random": "随机",
             "graph_block_noise": "屏蔽噪声源（诊断）", "graph_diverse": "Graph 多样性选择"}
    lines = ["# 附录 E：稳定性与噪声诊断", "",
        "## 范围与方法", "",
        "沿用附录 D 的 CIFAR-10 冻结 ResNet18 分类头实验：3 个数据 seed x 3 个训练 seed，1,000 样本/数据集，其中 100 个错标。仅复用既有五 epoch 真实轨迹，不重新训练、不调参、不改动旧分数。不是 Table 5 全网络实验。", "",
        "本轮完成排序矩阵、原始/标准化分数分布、逐 seed 小提琴图、噪声传播与反事实、1/2/5/10/20/40% Top-k 污染率、标签/梯度一致性及度/密度分桶、成功与失败极端邻域案例。", "",
        "真实 CIFAR 标签仅用于核验错标及事后诊断。实际 gate 仍使用原实验的观测标签与初始化梯度，不使用真实标签。", "",
        "## 排序稳定性", "",
        "同一数据集内比较训练 seed 的分数，禁止将不同数据集的局部索引直接对应。每个数据 seed 先平均三个训练 seed 对的 Spearman，再报告三个数据 seed 上的均值（标准差）；相关系数不是独立重复，不能据此进行独立样本显著性推断。", "",
        "| 方法 | 平均训练 seed 排序相关 |",
        "|---|---:|"]
    for method in names:
        group = [float(r["mean_pairwise_spearman"]) for r in stability if r["method"] == method]
        if group:
            lines.append("| %s | %s |" % (names[method], fmt(group)))
    lines += ["", "完整 21x21 矩阵分别保存每个数据 seed 的七方法 x 三训练 seed 相关关系。小提琴图展示 raw/Graph、正确/错误标签四组分布，分数仅在同一方法和 seed 内标准化；原始数值另存，不能把标准化后的幅度当成原始贡献大小。", "",
        "## Top-k 污染与识别", "",
        "下表为 20% 保留预算；完整六个预算均保留。均值（样本标准差）来自九个数据/训练交叉单元，不是九个独立数据集。", "",
        "| 方法 | 所选污染率 | 噪声 AUROC |", "|---|---:|---:|"]
    for method in names:
        group = [r for r in contamination if r["method"] == method and float(r["budget"]) == .2]
        lines.append("| %s | %s | %s |" % (names[method], fmt([float(r["contamination"]) for r in group]), fmt([float(r["noise_auroc"]) for r in group])))
    lines += ["", "## 噪声传播", "",
        "边的方向定义为邻居 source 向当前样本 recipient 发送消息，即分数更新式中的 j -> i。分别保存语义权重和 gate 权重乘置信度，不将归一化后的边权比例误当成实际传播质量。", "",
        "| 接收样本 | 语义图中噪声源权重占比 | 门控后噪声源权重占比 | 门控后平均总入边权重 |", "|---|---:|---:|---:|"]
    for recipient in (0, 1):
        fractions = []
        for key in ("semantic_weight", "gated_effective_weight"):
            fractions.append(np.mean([r["share_within_recipient_type"] for r in flow if r["recipient_corrupted"] == recipient and r["source_corrupted"] == 1 and r["weighting"] == key]))
        mass = sum(r["mean_mass_per_recipient"] for r in flow if r["recipient_corrupted"] == recipient and r["weighting"] == "gated_effective_weight") / 9
        lines.append("| %s | %.5f | %.5f | %.5f |" % ("错标" if recipient else "正确标签", *fractions, mass))
    lines += ["", "| 样本组 | 被 Graph 提升排名的比例 | 平均排名百分位变化 | 屏蔽噪声源后的平均百分位变化 |", "|---|---:|---:|---:|"]
    for status in (0, 1):
        group = [r for r in effects if int(r["corrupted"]) == status]
        lines.append("| %s | %.5f | %.5f | %.5f |" % ("错标" if status else "正确标签",
            *[np.mean([float(r[k]) for r in group]) for k in ("promoted_fraction", "mean_percentile_lift", "mean_blocked_rank_change")]))
    lines += ["", "历史项与 pair 项均拆分为正确/错标来源。分解固定实际总消息的标准差分母，对两个来源分别中心化，因而相加重建实际修正。它是带批内中心化的代数分配，不是错误标签的独立因果效应；中心化可影响没有直接噪声入边的样本。", "",
        "反事实屏蔽已知错标节点发出的历史和批内消息，再重新计算尺度匹配；不改变原始分数、模型轨迹、图、gate 或标签。它使用事后错标信息，只是 oracle 诊断，不是可部署评分方法，也不是删除/纠正噪声后重新训练的效果。屏蔽零个源与实际路径一致；屏蔽全部源退化为原始分数，测试通过。", "",
        "## 邻域分桶", "",
        "对观测标签一致性、带符号的梯度余弦一致性、对称图正边度数、未归一化近邻余弦均值分别按五分位分桶。重合边界合并，所以可能少于五桶。密度不是恒定的归一化边权均值。每桶保存样本数、错标数、Top-20% 所选数量与污染率、平均排名变化及桶内 AUROC；若只有一种标签，AUROC 留空，不伪造数值。", "",
        "所有九个 seed 单元都保存。分桶图固定展示 train seed 0 的三个数据 seed，不是从结果中挑选 seed；degree/density/标签一致性只由数据和图决定，梯度一致性依赖初始化。", "",
        "## 成功与失败案例", "",
        "固定 data seed 0 / train seed 0，在错标和正确标签样本中分别选择排名百分位变化的最小/最大者，展示原图与五条最高语义权重邻边。这是预先规定的极端案例规则，不代表出现频率；正确标签样本的晋升也不等于已经证明其训练效用。", "",
        "| 案例 | CIFAR 训练源 ID | 真实 / 观测标签 | 原始排名 -> Graph 排名 |", "|---|---:|---|---:|"]
    cases = json.loads((OUT / "cases.json").read_text())
    for r in cases:
        lines.append("| %s | %d | %s / %s | %d -> %d |" % (r["title"], r["source_id"], r["true_label"], r["observed_label"], r["raw_rank"], r["graph_rank"]))
    lines += ["", "## 边界与产物", "",
        "本轮不能证明 Graph 全面提升子集训练质量。附录 D 中默认 Graph 的 20% 子集准确率低于随机，该不利结果仍保留。噪声识别改善、排序稳定性与下游训练质量必须分开陈述。", "",
        "完整数值见 `results/`：nodes/edges/score_distribution/rank_correlations/contamination/buckets/noise_flow/step_messages/propagation_effects CSV，及逐 seed 的反事实 NPZ、矩阵 NPZ、案例 JSON。所有图均有 PNG/PDF/SVG。源缓存和脚本哈希记录于 provenance.json。", "",
        "验证涵盖 9,000 个节点、90,000 条有向边、63,000 条分数记录、486 个污染率单元，分桶数目完整，原分数重建、消息分解、反事实端点及真实错标标记核验通过。未重新训练模型或修改原实验结果。"]
    (HERE / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
    for ax, key in zip(axes, ("semantic_weight", "gated_effective_weight")):
        matrix = np.array([[np.mean([r["share_within_recipient_type"] for r in flow if r["weighting"] == key and r["recipient_corrupted"] == i and r["source_corrupted"] == j]) for j in (0, 1)] for i in (0, 1)])
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, "%.2f%%" % (100 * value), ha="center", va="center", color="white" if value > .5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Clean", "Corrupt"], yticklabels=["Clean", "Corrupt"],
            xlabel="Message source", ylabel="Message recipient", title=key.replace("_", " "))
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("noise_flow." + suffix), dpi=180)
    plt.close(fig)
    (OUT / "audit.json").write_text(json.dumps(dict(status="passed", source_hashes="unchanged",
        nodes=9000, edges=90000, distributions=63000, contamination_cells=486,
        edge_orientation="neighbor source -> recipient", bucket_counts="complete",
        stored_topk_contamination="verified"), indent=2) + "\n")
    print("Independent audit passed; report and noise-flow plot written.")


if __name__ == "__main__":
    main()
