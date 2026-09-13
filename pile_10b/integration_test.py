"""Two-process CPU test: interrupted+resumed run must match uninterrupted run."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[1]
    parent = Path(tempfile.mkdtemp(prefix="pile10b-integration-"))
    fixture = parent / "fixture"
    subprocess.run([sys.executable, "-m", "pile_10b.smoke_fixture", str(fixture)], cwd=root, check=True)
    env = dict(os.environ, PILE10B_TEST_CPU="1", GPUS="2", SKIP_INSTALL="1",
               PYTHON=sys.executable, CONFIG=str(fixture / "config.json"), OMP_NUM_THREADS="1")
    def launch(name, allow):
        run_env = dict(env, WORK_DIR=str(parent / name), ALLOW_LONG_RUN=str(int(allow)))
        with (parent / (name + ("-resume" if allow else "-stop") + ".log")).open("w") as f:
            return subprocess.run(["bash", "pile_10b/run.sh"], cwd=root, env=run_env, stdout=f, stderr=subprocess.STDOUT)
    stopped = launch("resumed", False)
    assert stopped.returncode != 0, "Budget guard must stop the run"
    assert (parent / "resumed/seed_0/checkpoint.pt").exists(), (parent / "resumed-stop.log").read_text()
    for name in ("resumed", "continuous"):
        result = launch(name, True)
        assert result.returncode == 0, (parent / (name + "-resume.log")).read_text()
    for method in ("adam1", "graph"):
        a = np.load(parent / ("resumed/seed_0/%s_scores.npy" % method))
        b = np.load(parent / ("continuous/seed_0/%s_scores.npy" % method))
        np.testing.assert_array_equal(a, b)
    a = torch.load(parent / "resumed/seed_0/checkpoint.pt", weights_only=True)
    b = torch.load(parent / "continuous/seed_0/checkpoint.pt", weights_only=True)
    for key in a["ln_f"]:
        torch.testing.assert_close(a["ln_f"][key], b["ln_f"][key], rtol=0, atol=0)
    print("PASS: two-process scores and final parameters are bitwise equal after resume")
    print("Test artifacts:", parent)


if __name__ == "__main__":
    main()
