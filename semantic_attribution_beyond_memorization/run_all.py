"""Run both local studies across three seeds on separate GPUs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

def worker(task, gpu):
    for seed in range(3):
        out = BASE / 'results' / task / f'seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        result = out / 'result.json'
        if result.exists():
            data = json.loads(result.read_text())
            expected_hash = hashlib.sha256((BASE / 'inputs/dataset.json').read_bytes()).hexdigest()
            if data['seed'] != seed or data['task'] != task or data['data_sha256'] != expected_hash:
                raise ValueError('Incompatible existing result')
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        print(f'Start {task} seed={seed} GPU={gpu}', flush=True)
        with (out / 'run.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'semantic_attribution_beyond_memorization.run_study',
                            '--task', task, '--seed', str(seed)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Done {task} seed={seed}', flush=True)

def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, 'paraphrase', 0), pool.submit(worker, 'dynamics', 1)]
        for future in futures:
            future.result()

if __name__ == '__main__':
    main()
