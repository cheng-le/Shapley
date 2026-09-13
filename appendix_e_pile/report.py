"""Independent audit and Chinese report for the Pile Appendix E."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
SOURCE = ROOT / "efficiency_scalability_ablations/second_order_online"


def read(name):
    with (OUT / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / (name + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(values, digits=5):
    return "%.*f (%.*f)" % (digits, np.mean(values), digits, np.std(values, ddof=1))


def main():
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest() == provenance["script_sha256"]
    for path, digest in provenance["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    nodes, edges, bins, contamination, stability, effects = [read(n) for n in (
        "nodes", "edges", "buckets", "contamination", "stability_pairs", "propagation_effects")]
    assert len(nodes) == 3000 and len(edges) == 30000 and len(contamination) == 108
    assert len(read("score_distribution")) == 12000 and len(read("rank_correlations")) == 288
    for seed in range(3):
        local = [r for r in nodes if int(r["seed"]) == seed]
        assert len(local) == 1000 and sum(int(r["corrupted"]) for r in local) == 100
        by_id = {int(r["local_id"]): r for r in local}
        totals = np.zeros(1000)
        for r in [r for r in edges if int(r["seed"]) == seed]:
            assert r["source_corrupted"] == by_id[int(r["source"])]["corrupted"]
            assert r["recipient_corrupted"] == by_id[int(r["recipient"])]["corrupted"]
            assert int(r["source"]) != int(r["recipient"])
            totals[int(r["recipient"])] += float(r["weight"])
        np.testing.assert_allclose(totals, 1, atol=1e-6)
        for variable in ("domain_agreement", "initial_gradient_agreement", "degree", "density"):
            group = [r for r in bins if int(r["seed"]) == seed and r["variable"] == variable]
            assert sum(int(r["count"]) for r in group) == 1000
            assert sum(int(r["corrupted_count"]) for r in group) == 100
        values = np.array([float(by_id[i]["graph_score"]) for i in range(1000)])
        for row in [r for r in contamination if int(r["seed"]) == seed and r["method"] == "graph"]:
            chosen = np.argsort(-values, kind="stable")[:int(row["count"])]
            assert sum(int(by_id[int(i)]["corrupted"]) for i in chosen) == int(row["corrupted_selected"])
        with np.load(OUT / ("initial_diagnostics_seed%d.npz" % seed)) as z:
            assert z["features"].shape == (1000, 768) and z["gradients"].shape == (1000, 1536)
            assert np.isfinite(z["gradients"]).all()
            np.testing.assert_allclose(np.linalg.norm(z["features"], axis=1), 1, atol=1e-5)
    summary = []
    names = {"inrun2": "二阶 In-Run", "adam2": "二阶 Adam-aware", "adam1": "一阶 Adam-aware", "graph": "Graph，无 HVP",
             "graph_diverse": "Graph 多样性选择", "block_noise": "屏蔽破坏源（oracle 诊断）"}
    for method in names:
        for budget in (.01, .02, .05, .1, .2, .4):
            group = [r for r in contamination if r["method"] == method and float(r["budget"]) == budget]
            summary.append(dict(method=method, budget=budget,
                contamination_mean=float(np.mean([float(r["contamination"]) for r in group])),
                contamination_std=float(np.std([float(r["contamination"]) for r in group], ddof=1)),
                noise_auroc_mean=float(np.mean([float(r["noise_auroc"]) for r in group])),
                noise_auroc_std=float(np.std([float(r["noise_auroc"]) for r in group], ddof=1))))
    write("contamination_summary", summary)
    flow = []
    for seed in range(3):
        for recipient in (0, 1):
            group = [r for r in edges if int(r["seed"]) == seed and int(r["recipient_corrupted"]) == recipient]
            mass = sum(float(r["weight"]) for r in group)
            corrupted = sum(float(r["weight"]) for r in group if int(r["source_corrupted"]))
            flow.append(dict(seed=seed, recipient_corrupted=recipient, noise_source_weight_share=corrupted / mass))
    write("noise_flow", flow)
    lines = ["# 附录 E：Pile 稳定性与噪声诊断", "",
        "## 范围", "",
        "本节按用户要求仅使用 **Pile**。以 `efficiency_scalability_ablations/second_order_online` 的 GPT-2 / 1k Pile 在线实验为基准：三种子、128-token 输入、只训练最终 LayerNorm，Graph 使用一阶 Adam-aware 分数及语义图交互，源记录确认其 HVP 调用为零。不是旧 Graph-v2 的二阶缓存后处理，也不是 100k 全量诊断。", "",
        "不重新训练、不重新调参、不改变已有分数。仅对相同污染输入补算预训练初始状态、eval 模式下的语义表示和 1,536 维逐样本梯度，用于邻域诊断；这些梯度不输入 Graph 评分。", "",
        "## 排序矩阵与分布", "",
        "每种情景分别保存四方法 x 三种子的 12x12 Spearman 矩阵。不同种子按相同原始语料 ID 对齐。clean 情景仅改变训练 seed；noise 情景同时改变训练随机性和被破坏样本/内容，因此后者不是纯训练 seed 稳定性。", "",
        "| 方法 | clean 跨 seed Spearman | noise 跨 seed Spearman |", "|---|---:|---:|"]
    for method in ("inrun2", "adam2", "adam1", "graph"):
        vals = [[float(r["spearman"]) for r in stability if r["scenario"] == scenario and r["method"] == method] for scenario in ("clean", "noise")]
        lines.append("| %s | %s | %s |" % (names[method], fmt(vals[0]), fmt(vals[1])))
    lines += ["", "括号是三个 seed 对相关系数的描述性标准差，三对共享 seed，不是独立重复或置信区间。", "",
        "[排序矩阵](results/rank_matrices.png)；[逐 seed 小提琴图](results/per_seed_distributions.png)。小提琴图的 clean/noise 指同一污染训练轨迹内未被破坏/被破坏的样本，不是 clean/noise 两次训练。每个方法/seed 内单独 z 标准化，原始分数完整保留于 CSV。", "",
        "## Top-k 污染率", "",
        "完整保留 1/2/5/10/20/40% 六个预算。以下为 10% 预算，均值（样本标准差）来自三个种子。原始数据破坏率为 10%，AUROC 用负贡献识别被破坏样本。", "",
        "| 方法 | 所选破坏比例 | 噪声 AUROC |", "|---|---:|---:|"]
    for method in names:
        r = next(r for r in summary if r["method"] == method and r["budget"] == .1)
        lines.append("| %s | %.5f (%.5f) | %.5f (%.5f) |" % (names[method], r["contamination_mean"], r["contamination_std"], r["noise_auroc_mean"], r["noise_auroc_std"]))
    lines += ["", "[完整污染曲线](results/topk_contamination.png)，[全部预算原始结果](results/contamination.csv)。oracle 行使用事后破坏标记，只作诊断，不能当成新方法效果。", "",
        "## 消息传播与反事实", "",
        "边定义为邻居 source j 向 recipient i 发出消息。当前 Pile Graph 不使用梯度门控，所以只报告其实际语义边，不凭空添加 gate。", "",
        "| 接收样本组 | 破坏来源的平均入边权重占比 | 排名被提升比例 | 平均排名百分位变化 | 屏蔽破坏源后平均百分位变化 |", "|---|---:|---:|---:|---:|"]
    for status in (0, 1):
        group = [r for r in effects if int(r["corrupted"]) == status]
        lines.append("| %s | %.5f | %.5f | %.5f | %.5f |" % ("被破坏" if status else "未破坏",
            np.mean([r["noise_source_weight_share"] for r in flow if r["recipient_corrupted"] == status]),
            *[np.mean([float(r[k]) for r in group]) for k in ("promoted_fraction", "mean_percentile_lift", "blocked_percentile_delta")]))
    lines += ["", "[真实消息随 step 的变化](results/noise_messages.png)。分别记录历史与批内 pair 的未破坏/破坏来源；其绝对消息量不是下游性能影响量。", "",
        "分解固定实际总项的标准差分母，分别中心化两类来源，再相加重建真实修正。它包含批内中心化耦合，是代数分配而非独立因果贡献。反事实将破坏节点发出的历史/pair 消息置零，并重新计算尺度匹配；原始贡献、训练轨迹、图和文本不变。它不是去噪重训，也不能证明腐坏造成了多少训练损失。", "",
        "重建分数与原始 Graph 缓存一致；屏蔽零个源等于原路径，屏蔽全部源退化为一阶分数，均已验证。", "",
        "## 邻域一致性与分桶", "",
        "Pile 无 CIFAR 式类别标签，因此以 **原始语料来源 domain 一致性**替代 neighbor label agreement，不虚构类别正确率。破坏文本保留其来源域元数据，但该域不被当作混合破坏文本的真实语义标签。", "",
        "另外报告初始逐样本梯度的带符号加权余弦一致性、对称图正边度数，以及近邻原始余弦均值作为密度。密度不是恒为 1/k 的归一化边权均值。梯度来自初始 eval 模型，不是训练中梯度一致性的测量。", "",
        "四指标按 seed 内五分位分桶，重复边界合并；每桶保存人数、破坏数、Top-20% 所选人数/破坏率、平均排名变化和桶内 AUROC。单一类别桶的 AUROC 留空。分桶是描述性关联，不能直接推出因果机制。", "",
        "[分桶图](results/neighborhood_buckets.png)，[全部桶和分母](results/buckets.csv)。", "",
        "## Pile 语料邻域案例", "",
        "固定 seed 0，分别取破坏/未破坏样本排名变化的最小与最大者。这是极端案例规则，不代表典型发生频率，未破坏样本晋升也不必然代表更高训练效用。展示实际不超过 128-token 的输入、干净对应输入、五条最高语义权重邻边以及实际 16 条归因查询中余弦最接近的查询；查询匹配不等于因果归因。", "",
        "| 案例 | 原始训练语料 ID | 来源域 | 原始排名 -> Graph 排名 |", "|---|---:|---|---:|"]
    for r in json.loads((OUT / "cases.json").read_text()):
        lines.append("| %s | %d | %s | %d -> %d |" % (r["title"], r["source_id"], r["original_domain"], r["raw_rank"], r["graph_rank"]))
    lines += ["", "[全部实际语料与邻居](results/cases.md)，[可复核案例 JSON](results/cases.json)。", "",
        "## 限制与验证", "",
        "原在线实验噪声 AUROC 接近 0.5，本轮不会把微小变化宣称为强噪声识别。原子集微调未优于预训练参考的结果仍保留；稳定性或邻域支持不能替代下游效用。源实验已冻结，本轮不根据诊断重新选参。", "",
        "本节的最终产物仅在 `appendix_e_pile/`。先前启动的 CIFAR 诊断仅保留为草稿，不纳入本附录结论。", "",
        "[独立审计](results/audit.json)验证源缓存哈希、3,000 节点、30,000 边、12,000 分数记录、108 个污染率单元、分桶人数、边方向和 Top-k 计数。图均提供 PNG/PDF/SVG，原始表和反事实数组完整保存。"]
    (HERE / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    audit = dict(status="passed", source_hashes="unchanged", nodes=3000, edges=30000,
        distributions=12000, contamination_cells=108, bucket_counts="complete",
        graph_topk_counts="verified", diagnostic_gradient_shape=[1000, 1536])
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
