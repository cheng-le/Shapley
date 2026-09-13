import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pile_10b import pipeline as p


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, **kwargs):
        return list(range(16))


def config():
    c = json.loads(Path(__file__).with_name("config.json").read_text())
    c.update(train_tokens=126, val_tokens=30, sequence_length=8,
             query_blocks=2, graph_k=3, feature_dim=4, graph_backend="exact")
    return c


def fixture_rows(seed=2027):
    found = dict(train=[], val=[], query=[])
    for i in range(100000):
        text = "document %d science language and experiments" % i
        h = hashlib.sha256(text.encode()).hexdigest()
        b = int(hashlib.sha256((str(seed) + h).encode()).hexdigest()[:16], 16) % 1000
        split = "val" if b == 0 else ("query" if b == 1 else "train")
        if len(found[split]) < 20:
            found[split].append(dict(text=text, meta=dict(pile_set_name=split)))
        if all(len(v) >= 20 for v in found.values()):
            break
    return found["train"] + found["query"] + found["val"]


class PipelineTests(unittest.TestCase):
    def test_exact_budget_split_dedup_and_reuse(self):
        c = config()
        source = fixture_rows()
        source = [source[0]] * 5 + source
        with tempfile.TemporaryDirectory() as path:
            work = Path(path)
            with patch.object(p, "tokenizer", return_value=FakeTokenizer()), patch.object(p, "rows", return_value=iter(source)):
                p.prepare(c, work)
            data = json.loads((work / "data.json").read_text())
            self.assertEqual((data["train"], data["val"], data["query"]), (126, 30, 16))
            con = p.sqlite3.connect(str(work / "documents.sqlite"))
            hashes = con.execute("SELECT hash, count(*) FROM docs GROUP BY hash").fetchall()
            self.assertTrue(all(count == 1 for _, count in hashes))
            for split in ("train", "val", "query"):
                domains = con.execute("SELECT DISTINCT domain FROM docs WHERE split=?", (split,)).fetchall()
                self.assertEqual(domains, [(split,)])
            con.close()
            with patch.object(p, "rows", side_effect=AssertionError("must reuse")):
                p.prepare(c, work)

    def test_extraction_resume_after_exhaustion(self):
        c, source = config(), fixture_rows()
        with tempfile.TemporaryDirectory() as path:
            work = Path(path)
            with patch.object(p, "tokenizer", return_value=FakeTokenizer()), patch.object(p, "rows", return_value=iter(source[:4])):
                with self.assertRaises(RuntimeError):
                    p.prepare(c, work)
            with patch.object(p, "tokenizer", return_value=FakeTokenizer()), patch.object(p, "rows", return_value=iter(source)):
                p.prepare(c, work)
            self.assertEqual((work / "train.bin").stat().st_size, 126 * 4)

    def test_graph_matches_reference(self):
        import ast
        source = Path(__file__).resolve().parents[1] / "efficiency_scalability_ablations/second_order_online/run.py"
        tree = ast.parse(source.read_text())
        node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "OnlineGraph")
        scope = {"np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        nn = np.array([[1, 2], [0, 2], [1, 0], [0, 2]])
        weights = np.ones_like(nn, dtype=np.float32) / 2
        ref = scope["OnlineGraph"](nn, weights)
        history = np.zeros(4, np.float32)
        for ids, values in (([0, 1], [1., 2.]), ([2, 3], [-1., 3.])):
            ids, values = np.array(ids), np.array(values, np.float32)
            actual = p.graph_update(ids, values, history, nn, weights, .2)
            np.testing.assert_allclose(actual, ref.update(ids, values), rtol=1e-6)
        np.testing.assert_array_equal(history, ref.history)

    def test_exact_neighbors_and_permutation(self):
        f = np.random.default_rng(3).normal(size=(21, 4)).astype(np.float32)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        nn, _ = p.exact_neighbors(f, np.arange(21), 3)
        sim = f @ f.T
        np.fill_diagonal(sim, -np.inf)
        expected = np.argsort(-sim, axis=1)[:, :3]
        for a, b in zip(nn, expected):
            self.assertEqual(set(a), set(b))
        for n in (16, 17, 100):
            a, offset = p.permutation(n, 8)
            self.assertEqual(len(set((a * np.arange(n) + offset) % n)), n)

    def test_config_lock(self):
        c = config()
        with tempfile.TemporaryDirectory() as path:
            work = Path(path)
            p.lock_config(c, work)
            changed = copy.deepcopy(c)
            changed["alpha"] = .4
            with self.assertRaises(RuntimeError):
                p.lock_config(changed, work)
        c["train_tokens"] = 129
        with self.assertRaises(ValueError):
            p.validate(c)

    def test_hnsw_graph(self):
        try:
            import hnswlib
        except ImportError:
            self.skipTest("Install deployment requirements to test HNSW")
        c = config()
        c["graph_backend"] = "hnsw"
        f = np.random.default_rng(9).normal(size=(p.blocks(c), c["feature_dim"])).astype(np.float32)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as path:
            work = Path(path)
            f.tofile(work / "features.bin")
            p.dump(work / "features.json", dict(sha256=p.digest_file(work / "features.bin")))
            p.graph(c, work)
            result = json.loads((work / "graph.json").read_text())
            self.assertGreaterEqual(result["recall"], .99)
            nn = np.memmap(work / "neighbors.bin", dtype="int32", shape=(len(f), c["graph_k"]))
            self.assertFalse(np.any(nn == np.arange(len(f))[:, None]))

    def test_artifact_corruption_detection(self):
        with tempfile.TemporaryDirectory() as path:
            work = Path(path)
            for name in ("train", "val", "query", "features", "neighbors", "weights"):
                (work / (name + ".bin")).write_bytes(b"sample")
            h = p.digest_file(work / "train.bin")
            p.dump(work / "data.json", dict(sha256={name: h for name in ("train", "val", "query")}))
            p.dump(work / "features.json", dict(sha256=h))
            p.dump(work / "graph.json", dict(hashes={name: h for name in ("neighbors", "weights")}))
            p.verify(config(), work)
            (work / "weights.bin").write_bytes(b"broken")
            with self.assertRaises(RuntimeError):
                p.verify(config(), work)


if __name__ == "__main__":
    unittest.main()
