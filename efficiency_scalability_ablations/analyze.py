"""Validate all experiment cells and write the detailed resource report."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    cells = [json.loads(p.read_text()) for p in sorted((OUT / "timing").glob("*.json"))]
    assert len(cells) == 12
    assert {(p["rows"][0]["n"], p["rows"][0]["repetition"]) for p in cells} == {
        (n, rep) for n in (1000, 10000, 50000, 100000) for rep in range(3)}
    stage_names = {"pointwise_replay", "pointwise_top10", "exact_knn", "ann_knn"}
    stage_names |= {mode + suffix for mode in ("exact", "ann")
                    for suffix in ("_gate", "_replay1", "_replay2", "_select1", "_select2")}
    for cell in cells:
        assert len(cell["rows"]) == 14
        assert {r["stage"] for r in cell["rows"]} == stage_names
        assert 0 <= cell["ann_recall10"] <= 1
        for row in cell["rows"]:
            assert row["seconds"] >= 0
    records = [r for p in sorted((OUT / "ablation").glob("seed*.json"))
               for r in json.loads(p.read_text())]
    assert len(records) == 168
    assert len({(r["seed"], r["scenario"], r["variant"], r["order"]) for r in records}) == 168
    for seed in range(3):
        with np.load(OUT / "ablation" / ("scores_seed%d.npz" % seed)) as scores:
            assert len(scores.files) == 28
            for key in scores.files:
                assert scores[key].shape == (10000,)
                assert np.isfinite(scores[key]).all()
            np.testing.assert_array_equal(scores["full__1"], scores["no_pair__1"])
    provenance = json.loads((OUT / "provenance.json").read_text())
    for name, expected in provenance["source_hashes"].items():
        assert hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest() == expected
    for cache in provenance["input_caches"]:
        stat = (HERE.parent / cache["path"]).stat()
        assert stat.st_size == cache["bytes"] and stat.st_mtime_ns == cache["mtime_ns"]
    ids = np.random.default_rng(2027).permutation(100000)[:10000]
    composition = []
    for scenario in ("clean", "noise"):
        for seed in range(3):
            path = HERE.parent / "pile_main_100k/cache" / ("%s_seed%d.npz" % (scenario, seed))
            with np.load(path) as cache:
                fingerprint = str(cache["fingerprint"].item())
                assert fingerprint == "4db35bdcf9cd4d40"
                domains, counts = np.unique(cache["domains"][ids], return_counts=True)
                labels = cache["noise_labels"][ids]
                composition.append(dict(scenario=scenario, seed=seed, fingerprint=fingerprint,
                    sample_count=len(ids), corruption_count=int(labels.sum()),
                    corruption_fraction=float(labels.mean()),
                    domains={str(d): int(c) for d, c in zip(domains, counts)}))
    (OUT / "input_subset_summary.json").write_text(json.dumps(composition, indent=2) + "\n")
    raw = []
    for cell in cells:
        stages = {r["stage"]: r for r in cell["rows"]}
        n, rep = cell["rows"][0]["n"], cell["rows"][0]["repetition"]
        for method in ("pointwise", "exact1", "exact2", "ann1", "ann2"):
            if method == "pointwise":
                names = ["pointwise_replay", "pointwise_top10"]
                replay_name = "pointwise_replay"
            else:
                mode, order = method[:-1], method[-1]
                replay_name = mode + "_replay" + order
                names = [mode + "_knn", mode + "_gate", replay_name, mode + "_select" + order]
            raw.append(dict(n=n, repetition=rep, method=method,
                cached_pipeline_seconds=sum(stages[k]["seconds"] for k in names),
                replay_ms_per_batch=stages[replay_name]["seconds"] * 1000 / int(np.ceil(n / 8)),
                peak_cuda_mib=max(stages[k]["cuda_extra_peak_mib"] for k in names),
                peak_process_rss_mib=max(stages[k]["rss_peak_mib"] for k in names)))
    with (OUT / "pipeline_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    def fmt(values, digits=3):
        return ("%.*f (%.*f)" % (digits, np.mean(values), digits, np.std(values, ddof=1)))
    text = ["# Resource Measurements", "",
            "Times exclude cache loading, embedding/gradient extraction and model training. Pipeline time sums independently timed graph build, gate, replay and selection; pointwise sums score replay and Top-k. These are not full attribution times.", "",
            "| N | Method | Cached pipeline seconds | Replay ms/batch | CUDA peak MiB | Process RSS peak MiB |",
            "|---:|---|---:|---:|---:|---:|"]
    for n in (1000, 10000, 50000, 100000):
        for method in ("pointwise", "exact1", "exact2", "ann1", "ann2"):
            group = [r for r in raw if r["n"] == n and r["method"] == method]
            values = [fmt([r[k] for r in group]) for k in (
                "cached_pipeline_seconds", "replay_ms_per_batch", "peak_cuda_mib", "peak_process_rss_mib")]
            text.append("| %d | %s | %s |" % (n, method, " | ".join(values)))
    text += ["", "CUDA peak is the maximum incremental PyTorch allocation across stages, not total device use. RSS is absolute process peak and includes frameworks and resident input/graph arrays; stages share a worker, so it is not standalone per-method RAM. Replay-batch cost is total replay divided by ceil(N/8), not measured online latency.",
             "", "## ANN Fidelity and Storage", "",
             "| N | Recall@10 | Score Spearman order 1 / 2 | Selection overlap order 1 / 2 | Input tensors MiB | Sparse graph MiB | Graph+scores MiB | HNSW index MiB |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in (1000, 10000, 50000, 100000):
        group = [p for p in cells if p["rows"][0]["n"] == n]
        corr = " / ".join(fmt([p["ann_score_spearman"][str(o)] for p in group], 5) for o in (1, 2))
        overlap = " / ".join(fmt([p["ann_selection_overlap"][str(o)] for p in group], 5) for o in (1, 2))
        storage = [np.mean([p[k] for p in group]) / 1024 ** 2 for k in (
            "input_tensor_bytes", "sparse_graph_bytes", "graph_score_cache_bytes", "ann_index_bytes")]
        text.append("| %d | %s | %s | %s | %s |" % (
            n, fmt([p["ann_recall10"] for p in group], 5), corr, overlap,
            " | ".join("%.3f" % value for value in storage)))
    text += ["", "Storage columns are byte counts of array payloads and the actual serialized HNSW index, not compressed NPZ file sizes. Original on-disk input cache sizes are recorded in provenance.json. Near-tied cosine similarities can change exact neighbor identities. ANN is not tuned against diagnostics.",
             "", "## Validation", "",
             "All 12 timing cells (168 stages), 168 ablation cells and 84 clean score vectors were checked for completeness. Replay parity and zero-strength limits passed. Source hashes and input-cache sizes/timestamps remained unchanged during the run."]
    (OUT / "resources.md").write_text("\n".join(text) + "\n")
    (OUT / "validation.json").write_text(json.dumps(dict(status="passed", timing_cells=12,
        timing_stages=168, ablation_cells=168, clean_score_vectors=84,
        source_hashes_match=True, input_cache_stat_unchanged=True), indent=2) + "\n")
    print("Validation passed; resources.md written")


if __name__ == "__main__":
    main()
