"""Run missing cells with one isolated training process per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    folder = ROOT / 'diverse_data_selection/results'
    cells = json.loads((folder / 'pending_evaluations.json').read_text())
    raw = folder / 'raw'
    raw.mkdir(exist_ok=True)
    def worker(slot, gpu):
        for cell in cells[slot::len(args.gpus)]:
            tag = Path(cell['selected_indices']).stem
            output = raw / f'{tag}.json'
            if output.exists():
                saved = json.loads(output.read_text())
                if all(saved.get(k) == v for k, v in cell.items()) and 'val_loss' in saved:
                    continue
                raise ValueError(f'Incompatible existing output: {output}')
            config = raw / f'{tag}.config.json'
            config.write_text(json.dumps(cell, indent=2))
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, '-m', 'diverse_data_selection.train_cell',
                       '--gpu', str(gpu), '--cell', str(config), '--output', str(output)]
            print(f'Start GPU {gpu}: {tag}', flush=True)
            with (raw / f'{tag}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'Done GPU {gpu}: {tag}', flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, slot, gpu) for slot, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()

if __name__ == '__main__':
    main()
