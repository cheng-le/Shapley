"""Streamed tokens, global approximate graph, and matched first-order attribution.

This is a packed-block, projected-ANN, final-LayerNorm scaling experiment,
not an exact reproduction of document-level Table 5 or full-network Shapley.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import numpy as np


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def blocks(c):
    return math.ceil(c["train_tokens"] / c["sequence_length"])


def cpu_test():
    return os.environ.get("PILE10B_TEST_CPU") == "1"


def validate(c):
    for key in ("train_tokens", "val_tokens", "sequence_length", "feature_dim",
                "feature_batch", "feature_chunk", "graph_k", "batch_per_gpu",
                "query_blocks", "checkpoint_steps", "eval_steps", "pilot_steps"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError("Expected positive integer: " + key)
    if c["sequence_length"] < 2 or c["train_tokens"] % c["sequence_length"] == 1:
        raise ValueError("Every block must have at least two tokens, including the tail")
    if c["val_tokens"] % c["sequence_length"] == 1:
        raise ValueError("Validation tail needs at least two tokens")
    if blocks(c) <= c["graph_k"]:
        raise ValueError("Need more training blocks than graph_k")
    if blocks(c) >= 2**31:
        raise ValueError("Block IDs exceed the int32 graph format")
    if not math.isfinite(c["alpha"]) or c["alpha"] < 0 or c["learning_rate"] <= 0:
        raise ValueError("Invalid alpha or learning rate")
    if c["train_scope"] != "ln_f":
        raise ValueError("Only audited final LayerNorm attribution is implemented")
    if c["precision"] not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if c["graph_backend"] not in ("hnsw", "exact"):
        raise ValueError("Unknown graph backend")
    if c["graph_backend"] == "exact" and blocks(c) > 20000:
        raise ValueError("Exact backend is restricted to small verification runs")


def lock_config(c, work):
    validate(c)
    work.mkdir(parents=True, exist_ok=True)
    path = work / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != c:
            raise RuntimeError("Config differs from this WORK_DIR; use a new WORK_DIR")
    else:
        dump(path, c)


def preflight(c, work, gpus):
    import psutil
    import shutil
    import torch
    from huggingface_hub import HfApi
    if cpu_test() and (c["train_tokens"] > 100000 or c["precision"] != "fp32"):
        raise ValueError("CPU test mode requires <=100k tokens and fp32")
    if not cpu_test() and (not torch.cuda.is_available() or torch.cuda.device_count() < gpus):
        raise RuntimeError("Insufficient visible CUDA GPUs; check driver/CUDA_VISIBLE_DEVICES")
    devices = []
    for i in range(0 if cpu_test() else gpus):
        with torch.cuda.device(i):
            if c["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 unsupported; configure fp32 in a new run")
            torch.ones(1, device="cuda").sum().item()
        devices.append(dict(name=torch.cuda.get_device_name(i),
                            bytes=torch.cuda.get_device_properties(i).total_memory))
    n = blocks(c)
    ram = n * (c["feature_dim"] * 4 + c["graph_m"] * 12 + 128) + (4 << 30)
    disk = (c["train_tokens"] + c["val_tokens"]) * 12 + n * (c["feature_dim"] * 4 + c["graph_k"] * 8 + 64) + (10 << 30)
    # These are conservative planning estimates, not OS-enforced reservations.
    if psutil.virtual_memory().available < ram:
        raise RuntimeError("Insufficient available host RAM; estimated requirement %.1f GiB" % (ram / 2**30))
    existing = (work / "provenance.json").exists()
    if not existing and shutil.disk_usage(work).free < disk:
        raise RuntimeError("Insufficient free disk; estimated requirement %.1f GiB" % (disk / 2**30))
    if not existing:
        api = HfApi()
        model_revision = c["model_revision"]
        if not Path(c["model"]).is_dir():
            model_revision = api.model_info(c["model"], revision=model_revision).sha
        dataset_revision = c["dataset_revision"]
        local = []
        if c["local_jsonl"]:
            for path in c["local_jsonl"]:
                p = Path(path).resolve()
                local.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest_file(p)))
        else:
            dataset_revision = api.dataset_info(c["dataset"], revision=dataset_revision).sha
        dump(work / "provenance.json", dict(model_revision=model_revision,
             dataset_revision=dataset_revision, local_sources=local,
             pipeline_sha256=digest_file(__file__), config=c))
    provenance = json.loads((work / "provenance.json").read_text())
    if provenance["pipeline_sha256"] != digest_file(__file__):
        raise RuntimeError("Pipeline changed; preserve this run and use a new WORK_DIR")
    for item in provenance["local_sources"]:
        if digest_file(item["path"]) != item["sha256"]:
            raise RuntimeError("Local source changed: " + item["path"])
    dump(work / "hardware.json", dict(gpus=devices, world_size=gpus,
         available_ram=psutil.virtual_memory().available, estimated_ram=ram,
         estimated_disk=disk, free_disk=shutil.disk_usage(work).free))
    print("Preflight passed:", json.loads((work / "hardware.json").read_text()), flush=True)


def tokenizer(c, work):
    from transformers import AutoTokenizer
    p = json.loads((work / "provenance.json").read_text())
    tok = AutoTokenizer.from_pretrained(c["model"], revision=p["model_revision"])
    tok.pad_token = tok.eos_token
    return tok


def rows(c, work):
    if c["local_jsonl"]:
        import gzip
        import io
        for path in c["local_jsonl"]:
            if path.endswith(".zst"):
                import zstandard
                with open(path, "rb") as raw:
                    with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                        with io.TextIOWrapper(reader, encoding="utf-8") as stream:
                            for line in stream:
                                yield json.loads(line)
            else:
                opener = gzip.open if path.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        yield json.loads(line)
    else:
        from datasets import load_dataset
        p = json.loads((work / "provenance.json").read_text())
        yield from load_dataset(c["dataset"], revision=p["dataset_revision"],
                                split=c["dataset_split"], streaming=True)


def prepare(c, work):
    if (work / "data.json").exists():
        for split in ("train", "val", "query"):
            expected = c["query_blocks"] * c["sequence_length"] if split == "query" else c[split + "_tokens"]
            if (work / (split + ".bin")).stat().st_size != expected * 4:
                raise RuntimeError("Token cache length mismatch")
        print("Reusing completed token cache", flush=True)
        return
    tok = tokenizer(c, work)
    con = sqlite3.connect(str(work / "documents.sqlite"))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS docs (hash TEXT PRIMARY KEY, scanned INTEGER, split TEXT, domain TEXT, tokens BLOB, count INTEGER)")
    con.execute("CREATE INDEX IF NOT EXISTS docs_split_scan ON docs(split,scanned)")
    con.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT)")
    value = con.execute("SELECT value FROM state WHERE id=1").fetchone()
    s = json.loads(value[0]) if value else dict(scanned=0, train=0, val=0, query=0)
    budgets = dict(train=c["train_tokens"], val=c["val_tokens"], query=c["query_blocks"] * c["sequence_length"])
    if any(s[key] < value for key, value in budgets.items()):
        for scanned, row in enumerate(rows(c, work), 1):
            if scanned <= s["scanned"]:
                continue
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("Expected string text field at row %d" % scanned)
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            bucket = int(hashlib.sha256((str(c["split_seed"]) + h).encode()).hexdigest()[:16], 16)
            split = "val" if bucket % 1000 == 0 else ("query" if bucket % 1000 == 1 else "train")
            remain = budgets[split] - s[split]
            if text.strip() and remain > 0 and not con.execute("SELECT 1 FROM docs WHERE hash=?", (h,)).fetchone():
                ids = tok.encode(text, add_special_tokens=False, truncation=False) + [tok.eos_token_id]
                ids = np.asarray(ids[:remain], dtype="<u4")
                meta = row.get("meta") or {}
                domain = str(meta.get("pile_set_name", "Unknown")) if isinstance(meta, dict) else "Unknown"
                con.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (h, scanned, split, domain, ids.tobytes(), len(ids)))
                s[split] += len(ids)
            s["scanned"] = scanned
            done = all(s[key] == value for key, value in budgets.items())
            if scanned % 1000 == 0 or done:
                con.execute("INSERT OR REPLACE INTO state VALUES (1,?)", (json.dumps(s),))
                con.commit()
                print("extraction", s, flush=True)
            if done:
                break
        else:
            con.execute("INSERT OR REPLACE INTO state VALUES (1,?)", (json.dumps(s),))
            con.commit()
            raise RuntimeError("Source exhausted before exact token budgets: " + str(s))
    hashes = {}
    for split in ("train", "val", "query"):
        path = work / (split + ".bin")
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for (blob,) in con.execute("SELECT tokens FROM docs WHERE split=? ORDER BY scanned", (split,)):
                f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        if tmp.stat().st_size != budgets[split] * 4:
            raise RuntimeError("Export token count mismatch")
        tmp.replace(path)
        hashes[split] = digest_file(path)
    domains = list(con.execute("SELECT split,domain,count(*),sum(count) FROM docs GROUP BY split,domain"))
    con.close()
    dump(work / "data.json", dict(**s, sha256=hashes, domains=domains,
         token_definition="GPT-2 token IDs including document EOS; no padding in budget",
         node_definition="packed fixed-length token block, may cross document boundaries",
         split="text SHA256 hash partition; exact duplicate texts cannot cross partitions"))


def distributed():
    import torch
    import torch.distributed as dist
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not cpu_test():
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("gloo" if cpu_test() else "nccl")
    return rank, world, torch.device("cpu") if cpu_test() else torch.device("cuda", local)


def barrier(world):
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def net_load(c, work, device):
    from transformers import GPT2LMHeadModel
    p = json.loads((work / "provenance.json").read_text())
    net = GPT2LMHeadModel.from_pretrained(c["model"], revision=p["model_revision"],
                                          attn_implementation="eager").to(device)
    if cpu_test() and net.config.n_embd > 64:
        raise ValueError("CPU integration mode is limited to tiny test models")
    if c["sequence_length"] > net.config.n_positions:
        raise ValueError("sequence_length exceeds model context")
    net.config.use_cache = False
    for param in net.parameters():
        param.requires_grad_(False)
    for param in net.transformer.ln_f.parameters():
        param.requires_grad_(True)
    return net


def amp(c):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if c["precision"] == "bf16" else contextlib.nullcontext()


def batch(tokens, indices, c, device, eos):
    import torch
    size = c["sequence_length"]
    ids = np.full((len(indices), size), eos, dtype=np.int64)
    mask = np.zeros_like(ids)
    for j, i in enumerate(indices):
        piece = tokens[int(i) * size:(int(i) + 1) * size]
        ids[j, :len(piece)], mask[j, :len(piece)] = piece, 1
    return {"input_ids": torch.as_tensor(ids, device=device), "attention_mask": torch.as_tensor(mask, device=device)}


def losses(net, b, c):
    import torch.nn.functional as F
    with amp(c):
        logits = net(**b).logits[:, :-1].float()
    target = b["input_ids"][:, 1:]
    mask = b["attention_mask"][:, 1:].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    return (loss.reshape(target.shape) * mask).sum(1) / mask.sum(1)


def features(c, work):
    import torch
    rank, world, device = distributed()
    if (work / "features.json").exists():
        if rank == 0:
            saved = json.loads((work / "features.json").read_text())
            if digest_file(work / "features.bin") != saved["sha256"]:
                raise RuntimeError("Completed feature cache hash mismatch")
        barrier(world)
        if world > 1:
            import torch.distributed as dist
            dist.destroy_process_group()
        return
    tok, net = tokenizer(c, work), net_load(c, work, device).eval()
    n, dim, chunk = blocks(c), c["feature_dim"], c["feature_chunk"]
    rng = np.random.default_rng(2027)
    projection = rng.normal(size=(net.config.n_embd, dim)).astype(np.float32) / math.sqrt(dim)
    projection = torch.as_tensor(projection, device=device)
    path = work / "features.bin"
    if rank == 0 and not path.exists():
        with path.open("wb") as f:
            f.truncate(n * dim * 4)
    barrier(world)
    if path.stat().st_size != n * dim * 4:
        raise RuntimeError("Feature cache shape mismatch")
    out = np.memmap(path, dtype="float32", mode="r+", shape=(n, dim))
    tokens = np.memmap(work / "train.bin", dtype="<u4", mode="r")
    done_dir = work / "feature_done"
    done_dir.mkdir(exist_ok=True)
    for start in range(rank * chunk, n, world * chunk):
        end = min(n, start + chunk)
        marker = done_dir / (str(start) + ".json")
        if marker.exists():
            saved = json.loads(marker.read_text())
            if hashlib.sha256(out[start:end].tobytes()).hexdigest() != saved["sha256"]:
                raise RuntimeError("Partial feature chunk hash mismatch")
            continue
        for left in range(start, end, c["feature_batch"]):
            right = min(end, left + c["feature_batch"])
            b = batch(tokens, range(left, right), c, device, tok.eos_token_id)
            with torch.no_grad(), amp(c):
                h = net.transformer(**b).last_hidden_state.float()
                mask = b["attention_mask"].unsqueeze(-1)
                v = (h * mask).sum(1) / mask.sum(1)
                v = torch.nn.functional.normalize(v @ projection, dim=1)
            out[left:right] = v.float().cpu().numpy()
        out.flush()
        dump(marker, dict(start=start, end=end, sha256=hashlib.sha256(out[start:end].tobytes()).hexdigest()))
        print("features rank=%d block=%d/%d" % (rank, end, n), flush=True)
    barrier(world)
    if rank == 0:
        dump(work / "features.json", dict(rows=n, dimension=dim, projection_seed=2027,
             sha256=digest_file(path), representation="initial GPT-2 mean hidden state, Gaussian projection, L2 normalization"))
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


def exact_neighbors(features, query_ids, k):
    # Chunk both query rows and database rows; never allocate an N x N matrix.
    q = np.asarray(features[query_ids])
    values = np.full((len(q), k), -np.inf, dtype=np.float32)
    indices = np.full((len(q), k), -1, dtype=np.int64)
    for start in range(0, len(features), 16384):
        sim = q @ np.asarray(features[start:start + 16384]).T
        for row, index in enumerate(query_ids):
            if start <= index < start + sim.shape[1]:
                sim[row, index - start] = -np.inf
        candidate_ids = np.broadcast_to(np.arange(start, start + sim.shape[1]), sim.shape)
        both = np.concatenate((values, sim), axis=1)
        ids = np.concatenate((indices, candidate_ids), axis=1)
        top = np.argpartition(-both, k - 1, axis=1)[:, :k]
        values, indices = np.take_along_axis(both, top, 1), np.take_along_axis(ids, top, 1)
    return indices, values


def graph(c, work):
    if (work / "graph.json").exists():
        print("Reusing completed global graph", flush=True)
        return
    n, dim, k = blocks(c), c["feature_dim"], c["graph_k"]
    if not (work / "features.json").exists():
        raise RuntimeError("Incomplete feature stage")
    f = np.memmap(work / "features.bin", dtype="float32", mode="r", shape=(n, dim))
    index = None
    if c["graph_backend"] == "hnsw":
        import hnswlib
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=c["graph_ef_construction"], M=c["graph_m"], random_seed=2027)
        index.set_num_threads(c["graph_threads"])
        index.set_ef(max(k + 1, c["graph_ef_search"]))
        for start in range(0, n, 16384):
            end = min(n, start + 16384)
            index.add_items(np.asarray(f[start:end]), np.arange(start, end))
            print("index %d/%d" % (end, n), flush=True)
    def query(ids):
        if index is None:
            return exact_neighbors(f, ids, k)
        nn, distance = index.knn_query(np.asarray(f[ids]), k=k + 1)
        out, sims = [], []
        for source, neighbors, distances in zip(ids, nn, distance):
            keep = neighbors != source
            out.append(neighbors[keep][:k])
            sims.append(1 - distances[keep][:k])
        return np.asarray(out), np.asarray(sims)
    sample = np.random.default_rng(2028).choice(n, min(c["graph_recall_queries"], n), replace=False)
    exact, _ = exact_neighbors(f, sample, k)
    approximate, _ = query(sample)
    recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(exact, approximate)]))
    dump(work / "ann_audit.json", dict(recall_at_k=recall, query_ids=sample.tolist(),
         scope="ANN recall in projected space, not original-space representation fidelity"))
    if recall < c["graph_min_recall"]:
        raise RuntimeError("ANN recall below configured threshold; create a new run with higher ef")
    nn = np.memmap(work / "neighbors.tmp", dtype="int32", mode="w+", shape=(n, k))
    ww = np.memmap(work / "weights.tmp", dtype="float32", mode="w+", shape=(n, k))
    for start in range(0, n, 1024):
        ids = np.arange(start, min(n, start + 1024))
        neighbors, similarities = query(ids)
        weight = np.maximum(similarities, 0)
        weight /= np.maximum(weight.sum(1, keepdims=True), 1e-8)
        if np.any(neighbors == ids[:, None]) or not np.isfinite(weight).all():
            raise AssertionError("Invalid semantic graph")
        nn[ids], ww[ids] = neighbors, weight
    nn.flush()
    ww.flush()
    del nn, ww
    (work / "neighbors.tmp").replace(work / "neighbors.bin")
    (work / "weights.tmp").replace(work / "weights.bin")
    dump(work / "graph.json", dict(nodes=n, k=k, backend=c["graph_backend"], recall=recall,
         hashes={name: digest_file(work / (name + ".bin")) for name in ("neighbors", "weights")}))


def graph_update(ids, raw, history, neighbors, weights, alpha):
    nn, ww = neighbors[ids], weights[ids]
    diffusion = (ww * history[nn]).sum(1)
    positions = {int(index): pos for pos, index in enumerate(ids)}
    pair = np.zeros_like(raw)
    for row in range(len(ids)):
        for neighbor, weight in zip(nn[row], ww[row]):
            pos = positions.get(int(neighbor))
            if pos is not None:
                pair[row] += weight * raw[pos]
    def match(term):
        a, b = raw.std(), term.std()
        return np.zeros_like(term) if a < 1e-12 or b < 1e-12 else (term - term.mean()) * (a / (b + 1e-12))
    corrected = raw + alpha * match(diffusion) + .5 * alpha * match(pair)
    history[ids] += raw
    return corrected


def verify(c, work):
    data = json.loads((work / "data.json").read_text())
    feature = json.loads((work / "features.json").read_text())
    edges = json.loads((work / "graph.json").read_text())
    expected = {name + ".bin": value for name, value in data["sha256"].items()}
    expected["features.bin"] = feature["sha256"]
    expected.update({name + ".bin": value for name, value in edges["hashes"].items()})
    for name, digest in expected.items():
        if digest_file(work / name) != digest:
            raise RuntimeError("Artifact hash mismatch: " + name)
    dump(work / "artifact_audit.json", dict(status="passed", sha256=expected))
    print("Token, feature and graph hashes verified", flush=True)


def permutation(n, seed):
    rng = np.random.default_rng(seed)
    a = int(rng.integers(1, n))
    while math.gcd(a, n) != 1:
        a = (a + 1) % n or 1
    return a, int(rng.integers(0, n))


def train(c, work):
    import torch
    import torch.distributed as dist
    rank, world, device = distributed()
    if not (work / "graph.json").exists():
        raise RuntimeError("Incomplete graph stage")
    tok = tokenizer(c, work)
    train_tokens = np.memmap(work / "train.bin", dtype="<u4", mode="r")
    val_tokens = np.memmap(work / "val.bin", dtype="<u4", mode="r")
    query_tokens = np.memmap(work / "query.bin", dtype="<u4", mode="r")
    n, local_b = blocks(c), c["batch_per_gpu"]
    global_b = world * local_b
    steps = math.ceil(n / global_b)
    nn = np.memmap(work / "neighbors.bin", dtype="int32", mode="r", shape=(n, c["graph_k"])) if rank == 0 else None
    ww = np.memmap(work / "weights.bin", dtype="float32", mode="r", shape=(n, c["graph_k"])) if rank == 0 else None
    def reduce(tensor):
        if world > 1:
            dist.all_reduce(tensor)
        return tensor
    def gather(tensor):
        if world == 1:
            return tensor
        parts = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(parts, tensor)
        return torch.cat(parts)
    for seed in c["seeds"]:
        out = work / ("seed_%d" % seed)
        out.mkdir(exist_ok=True)
        if (out / "complete.json").exists():
            continue
        torch.manual_seed(seed)
        net = net_load(c, work, device)
        params = tuple(net.transformer.ln_f.parameters())
        opt = torch.optim.AdamW(params, lr=c["learning_rate"], betas=(.9, .95), weight_decay=.1)
        checkpoint = out / "checkpoint.pt"
        start_step = 0
        raw_scores = np.zeros(n, np.float32) if rank == 0 else None
        graph_scores = np.zeros(n, np.float32) if rank == 0 else None
        baseline = None
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["world_size"] != world:
                raise RuntimeError("Resume must use the original GPU count")
            net.transformer.ln_f.load_state_dict(state["ln_f"])
            opt.load_state_dict(state["optimizer"])
            start_step, baseline = state["step"], state["baseline"]
            torch.set_rng_state(state["cpu_rng"][rank])
            if not cpu_test():
                torch.cuda.set_rng_state(state["cuda_rng"][rank], device)
            if rank == 0:
                raw_scores = state["raw"].numpy().copy()
                graph_scores = state["graph"].numpy().copy()
            del state
        else:
            torch.manual_seed(seed * 1009 + rank)
        a, offset = permutation(n, seed)
        def evaluate():
            net.eval()
            total = torch.zeros(2, dtype=torch.float64, device=device)
            count = math.ceil(len(val_tokens) / c["sequence_length"])
            with torch.no_grad():
                for i in range(rank, count, world):
                    b = batch(val_tokens, [i], c, device, tok.eos_token_id)
                    value = losses(net, b, c)[0]
                    targets = b["attention_mask"][:, 1:].sum()
                    total[0] += value.double() * targets
                    total[1] += targets
            reduce(total)
            return float((total[0] / total[1]).item())
        if baseline is None:
            baseline = evaluate()
        def save(step):
            cpu_rng = gather(torch.get_rng_state().to(device).unsqueeze(0)).cpu()
            rng = torch.get_rng_state() if cpu_test() else torch.cuda.get_rng_state(device)
            cuda_rng = gather(rng.to(device).unsqueeze(0)).cpu()
            if rank == 0:
                value = dict(step=step, world_size=world, baseline=baseline,
                    ln_f={k: v.detach().cpu() for k, v in net.transformer.ln_f.state_dict().items()},
                    optimizer=opt.state_dict(), cpu_rng=cpu_rng, cuda_rng=cuda_rng,
                    raw=torch.from_numpy(raw_scores), graph=torch.from_numpy(graph_scores))
                tmp = out / "checkpoint.tmp"
                with tmp.open("wb") as f:
                    torch.save(value, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(checkpoint)
            barrier(world)
        elapsed_steps = 0
        begin = time.monotonic()
        for step in range(start_step, steps):
            net.train()
            count = min(global_b, n - step * global_b)
            positions = np.arange(step * global_b, step * global_b + count)
            ids = (a * positions + offset) % n
            local_ids = ids[rank * local_b:(rank + 1) * local_b]
            # All ranks contribute to one utility gradient on the same parameters.
            query = torch.zeros(sum(p.numel() for p in params), device=device)
            for i in range(rank, c["query_blocks"], world):
                q = losses(net, batch(query_tokens, [i], c, device, tok.eos_token_id), c).mean()
                parts = torch.autograd.grad(q, params, create_graph=False)
                query += torch.cat([p.reshape(-1) for p in parts]).detach() / c["query_blocks"]
            reduce(query)
            raw = torch.zeros(local_b, device=device)
            gradients = torch.zeros_like(query)
            for j, i in enumerate(local_ids):
                b = batch(train_tokens, [i], c, device, tok.eos_token_id)
                parts = torch.autograd.grad(losses(net, b, c).mean(), params, create_graph=False)
                grad = torch.cat([p.reshape(-1) for p in parts]).detach()
                moment, variance = [], []
                for p in params:
                    s = opt.state.get(p, {})
                    moment.append(s.get("exp_avg", torch.zeros_like(p)).reshape(-1))
                    variance.append(s.get("exp_avg_sq", torch.zeros_like(p)).reshape(-1))
                m, v = torch.cat(moment), torch.cat(variance)
                t = step + 1
                direction = ((.9 * m + .1 * grad) / (1 - .9 ** t)) / (((.95 * v + .05 * grad.square()) / (1 - .95 ** t)).sqrt() + 1e-8)
                raw[j] = c["learning_rate"] / count * torch.dot(direction, query)
                # As in the reference runner, update uses a separate dropout draw.
                update = torch.autograd.grad(losses(net, b, c).mean(), params, create_graph=False)
                gradients += torch.cat([p.reshape(-1) for p in update]).detach() / count
            if not bool(torch.isfinite(raw).all() & torch.isfinite(gradients).all()):
                raise FloatingPointError("Non-finite attribution/update")
            global_raw = gather(raw).detach().cpu().numpy()[:count]
            reduce(gradients)
            if rank == 0:
                graph_scores[ids] = graph_update(ids, global_raw, raw_scores, nn, ww, c["alpha"])
            opt.zero_grad(set_to_none=True)
            cursor = 0
            for p in params:
                p.grad = gradients[cursor:cursor + p.numel()].view_as(p).clone()
                cursor += p.numel()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            elapsed_steps += 1
            if step % 10 == 0 and rank == 0:
                print("seed=%d step=%d/%d blocks=%d/%d" % (seed, step + 1, steps, min(n, (step + 1) * global_b), n), flush=True)
            if elapsed_steps == c["pilot_steps"]:
                seconds = time.monotonic() - begin
                hours = seconds / elapsed_steps * (steps - step - 1) / 3600
                estimate = torch.tensor(hours, device=device)
                if world > 1:
                    dist.all_reduce(estimate, op=dist.ReduceOp.MAX)
                hours = float(estimate.item())
                if rank == 0:
                    dump(out / "throughput.json", dict(measured_steps=elapsed_steps,
                        seconds=seconds, estimated_remaining_hours=hours,
                        excludes="future validation/checkpoint overhead; feature and graph stages already completed"))
                if hours > c["max_estimated_hours_per_seed"] and os.environ.get("ALLOW_LONG_RUN") != "1":
                    save(step + 1)
                    raise RuntimeError("Projected runtime exceeds budget. Inspect throughput.json; resume with ALLOW_LONG_RUN=1 to accept")
            if (step + 1) % c["eval_steps"] == 0:
                value = evaluate()
                if rank == 0:
                    dump(out / ("validation_%09d.json" % (step + 1)), dict(step=step + 1, validation_loss=value))
            if (step + 1) % c["checkpoint_steps"] == 0 or step + 1 == steps:
                save(step + 1)
        final = evaluate()
        if rank == 0:
            np.save(out / "adam1_scores.npy", raw_scores)
            np.save(out / "graph_scores.npy", graph_scores)
            dump(out / "complete.json", dict(seed=seed, steps=steps, blocks=n,
                input_tokens=c["train_tokens"], predicted_tokens=c["train_tokens"] - n,
                pretrained_validation_loss=baseline, final_validation_loss=final,
                hvp_calls=0, trainable_parameters=sum(p.numel() for p in params),
                comparison="paired scores on ONE shared training trajectory; not separate subset-trained models"))
        barrier(world)
        del net, opt
        torch.cuda.empty_cache()
    if world > 1:
        dist.destroy_process_group()


def report(c, work):
    results = []
    for seed in c["seeds"]:
        out = work / ("seed_%d" % seed)
        result = json.loads((out / "complete.json").read_text())
        for method in ("adam1", "graph"):
            scores = np.load(out / (method + "_scores.npy"), mmap_mode="r")
            if scores.shape != (blocks(c),) or not np.isfinite(scores).all():
                raise AssertionError("Incomplete/non-finite score array")
            result[method + "_score_mean"] = float(scores.mean(dtype=np.float64))
        results.append(result)
    values = [r["final_validation_loss"] for r in results]
    dump(work / "report.json", dict(status="complete", runs=results,
         final_validation_loss_mean=float(np.mean(values)),
         final_validation_loss_sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
         limitations=["10B token budget is not the entire Pile", "final LayerNorm only, not full-network training",
             "packed token-block valuation, not document valuation", "projected approximate global graph",
             "no corruption labels, noise AUROC/domain recall are not fabricated",
             "no HVP baselines or subset utility retraining in this scaling entrypoint"] ))
    print((work / "report.json").read_text(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("pile_10b/config.json"))
    p.add_argument("--work-dir", type=Path, default=Path("runs/pile_10b"))
    p.add_argument("stage", choices=("preflight", "prepare", "features", "graph", "verify", "train", "report"))
    p.add_argument("--gpus", type=int, default=4)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    if int(os.environ.get("RANK", 0)) == 0:
        lock_config(c, a.work_dir)
    else:
        validate(c)
    if a.stage == "preflight":
        preflight(c, a.work_dir, a.gpus)
    else:
        globals()[a.stage](c, a.work_dir)


if __name__ == "__main__":
    main()
