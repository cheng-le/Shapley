"""Check input/source hashes, actual derivative calls and utility evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"


def main():
    provenance = json.loads((OUT / "provenance.json").read_text())
    for path, digest in provenance["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    path = HERE / "data/manifest.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["data_manifest_sha256"]
    manifest = json.loads(path.read_text())
    for path, digest in manifest["tokens"].items():
        assert hashlib.sha256((HERE / "data" / path).read_bytes()).hexdigest() == digest
    assert len(set(manifest["train_indices"])) == 1000
    assert len(set(manifest["utility_val_indices"])) == 16
    assert len(set(manifest["evaluation_val_indices"])) == 128
    assert not (set(manifest["utility_val_indices"]) & set(manifest["evaluation_val_indices"]))
    cells = [json.loads(p.read_text()) for p in (OUT / "cells").glob("*.json")]
    assert len(cells) == 24
    for row in cells:
        second = row["method"] in ("inrun2", "adam2")
        assert row["audit"]["hvp_calls"] == (125 if second else 0)
        assert row["audit"]["create_graph_calls"] == (125 if second else 0)
        assert row["audit"]["grad_calls"] == (1250 if second else 1125)
        assert len(row["step_timings"]) == 125
        path = OUT / "cells" / ("%s_seed%d_%s.npz" % (row["scenario"], row["seed"], row["method"]))
        with np.load(path) as data:
            for key in ("scores", "first", "final_parameters", "training_losses"):
                assert np.isfinite(data[key]).all()
            assert data["scores"].shape == (1000,)
            if row["method"] == "graph":
                nn, ww = data["neighbors"], data["weights"]
                assert nn.shape == (1000, 10) and ww.shape == (1000, 10)
                assert not np.any(nn == np.arange(1000)[:, None])
                assert np.all(ww >= 0)
                np.testing.assert_allclose(ww.sum(1), 1, atol=1e-6)
    quality = [r for p in (OUT / "quality").glob("*.json") for r in json.loads(p.read_text())]
    assert len(quality) == 15
    for row in quality:
        assert len(set(row["selected_indices"])) == 100
        assert min(row["selected_indices"]) >= 0 and max(row["selected_indices"]) < 1000
        assert np.isfinite(row["validation_loss"])
    initial = [row["initial_validation_loss"] for row in quality]
    assert max(initial) == min(initial)
    report = dict(status="passed", input_and_source_hashes="unchanged", timing_cells=24,
        measured_training_steps=3000, utility_cells=15, utility_evaluation_disjoint=True,
        graph_measured_hvp_calls=0, graph_measured_create_graph_calls=0,
        baseline_measured_hvp_calls=sum(row["audit"]["hvp_calls"] for row in cells),
        common_pretrained_evaluation_loss=initial[0])
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
