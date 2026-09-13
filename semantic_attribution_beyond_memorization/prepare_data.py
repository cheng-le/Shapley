"""Freeze larger local case-study inputs from the authoritative Pile splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from table2_pile_memorization.corpora import GROUPS, CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'inputs'

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2027)
    targets = [{'text': g['original'], 'domain': g['domain'], 'sha256': sha(g['original'])} for g in GROUPS]
    excluded = {r['sha256'] for r in targets}
    reservoir, counts, per_domain = [], {}, {}
    train_hashes = set()
    with (ROOT / 'data/Pile_large/train.jsonl').open() as handle:
        accepted = 0
        for line in handle:
            raw = json.loads(line)
            text = raw['text']
            h = sha(text)
            if h in train_hashes:
                raise ValueError('Duplicate training document')
            train_hashes.add(h)
            if h in excluded:
                continue
            domain = raw['meta']['pile_set_name']
            record = {'text': text, 'domain': domain, 'sha256': h}
            accepted += 1
            if len(reservoir) < 2045:
                reservoir.append(record)
            else:
                position = int(rng.integers(accepted))
                if position < 2045:
                    reservoir[position] = record
            counts[domain] = counts.get(domain, 0) + 1
            pool = per_domain.setdefault(domain, [])
            if len(pool) < 128:
                pool.append(record)
            else:
                position = int(rng.integers(counts[domain]))
                if position < 128:
                    pool[position] = record
    validation = []
    with (ROOT / 'data/Pile_large/val.jsonl').open() as handle:
        for line in handle:
            raw = json.loads(line)
            if sha(raw['text']) in train_hashes:
                raise ValueError('Training/validation duplicate')
            if raw['meta']['pile_set_name'] == 'DM Mathematics':
                validation.append({'text': raw['text'], 'domain': 'DM Mathematics', 'sha256': sha(raw['text'])})
    indices = rng.choice(len(validation), size=16, replace=False)
    validation = [validation[int(i)] for i in indices]
    queries = [{'text': g[key], 'category': category, 'group': g['name'], 'target': index}
               for index, g in enumerate(GROUPS) for category, key in CATEGORIES]
    payload = {'sampling_seed': 2027, 'paraphrase_train': targets + reservoir,
               'queries': queries, 'dynamics_train': [r for d in sorted(per_domain) for r in per_domain[d]],
               'math_validation': validation}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + '\n'
    path = OUT / 'dataset.json'
    if path.exists() and path.read_text() != serialized:
        raise ValueError('Existing frozen inputs differ; use a new version directory')
    path.write_text(serialized)
    print(json.dumps({'paraphrase_count': len(payload['paraphrase_train']),
                      'dynamics_count': len(payload['dynamics_train']),
                      'domains': len(per_domain), 'math_validation': len(validation),
                      'sha256': sha(serialized)}, indent=2))

if __name__ == '__main__':
    main()
