"""Aggregate complete local semantic studies without selecting favorable cases."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from table2_pile_memorization.corpora import CATEGORIES

BASE = Path(__file__).resolve().parent

def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    out = BASE / 'results'
    expected_hash = hashlib.sha256((BASE / 'inputs/dataset.json').read_bytes()).hexdigest()
    data = {}
    for task in ('paraphrase', 'dynamics'):
        data[task] = []
        for seed in range(3):
            result = json.loads((out / task / f'seed{seed}/result.json').read_text())
            if result['seed'] != seed or result['task'] != task or result['data_sha256'] != expected_hash:
                raise ValueError('Mismatched source protocol')
            if len(result['losses']) != result['steps'] or not np.isfinite(result['losses']).all():
                raise ValueError('Incomplete training trajectory')
            data[task].append(result)
    rank_rows = [{**row, 'seed': seed} for seed, result in enumerate(data['paraphrase']) for row in result['ranks']]
    methods = list(dict.fromkeys(row['method'] for row in rank_rows))
    categories = [name for name, _ in CATEGORIES]
    summary = []
    for method in methods:
        for category in categories:
            item = {'method': method, 'category': category}
            for metric in ('rank', 'reciprocal_rank', 'hit_at_10'):
                runs = []
                for seed in range(3):
                    values = [r[metric] for r in rank_rows if (r['method'], r['category'], r['seed']) == (method, category, seed)]
                    if len(values) != 3:
                        raise ValueError('Incomplete case ranks')
                    runs.append(float(np.mean(values)))
                item[metric] = {'runs': runs, 'mean': float(np.mean(runs)), 'std': float(np.std(runs))}
            summary.append(item)
    write_csv(out / 'query_ranks.csv', rank_rows)
    lines = ['# Semantic Attribution: Local-Scale Study', '',
             '2,048 candidate documents, three fixed source groups, seeds 0/1/2. Lower source rank is better.',
             'Graph rows contain graph-only regularization; all methods are local implementations.', '',
             '| Method | ' + ' | '.join(categories) + ' |', '|---|' + '---:|' * 4]
    for method in methods:
        cells = []
        for category in categories:
            stats = next(r for r in summary if (r['method'], r['category']) == (method, category))['rank']
            cells.append(f'{stats["mean"]:.2f} ({stats["std"]:.3f})')
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(['', 'Similar-topic ranks are qualitative designated-source ranks, not unique-source retrieval accuracy.',
                  'These runs do not establish independence from pretraining memorization.', ''])
    history_rows = []
    for seed, result in enumerate(data['dynamics']):
        for snapshot in result['history']:
            for method, values in snapshot['methods'].items():
                for domain, value in values['signed_totals'].items():
                    history_rows.append({'seed': seed, 'step': snapshot['step'], 'method': method,
                                         'domain': domain, 'signed_total': value,
                                         'seen_item_mean': values['seen_item_means'][domain],
                                         'positive_share': values['positive_composition'][domain]})
    write_csv(out / 'domain_history.csv', history_rows)
    dynamic_methods = list(data['dynamics'][0]['history'][0]['methods'])
    math_summary = []
    lines.extend(['## Final Math-Domain Contribution', '',
                  '500 actual updates; 1,817 training documents and 16 separate math utility documents.', '',
                  '| Method | Signed total | Positive share (%) |', '|---|---:|---:|'])
    for method in dynamic_methods:
        runs = [r['history'][-1]['methods'][method] for r in data['dynamics']]
        totals = [r['signed_totals']['DM Mathematics'] for r in runs]
        shares = [r['positive_composition']['DM Mathematics'] for r in runs]
        math_summary.append({'method': method, 'signed_totals': totals, 'positive_shares': shares})
        lines.append(f'| {method} | {np.mean(totals):.3e} ({np.std(totals):.3e}) | {np.mean(shares):.3f} ({np.std(shares):.3f}) |')
    lines.extend(['', 'Positive shares are descriptive, not an accuracy metric. Signed totals for every domain are retained in domain_history.csv.', ''])
    (out / 'summary.md').write_text('\n'.join(lines))
    (out / 'summary.json').write_text(json.dumps({'paraphrase': summary, 'final_math': math_summary,
                                                 'data_sha256': expected_hash}, indent=2, allow_nan=False) + '\n')
    fig, ax = plt.subplots(figsize=(11, 5))
    for method in methods:
        group = [next(r for r in summary if (r['method'], r['category']) == (method, c))['rank'] for c in categories]
        ax.errorbar(range(4), [r['mean'] for r in group], yerr=[r['std'] for r in group],
                    marker='o', capsize=3, label=method)
    ax.set_yscale('log')
    ax.set_xticks(range(4), ['Partial exact', 'Paraphrase', 'Significant paraphrase', 'Similar topic'])
    ax.set_ylabel('Designated-source rank (log scale)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'paraphrase_ranks.{suffix}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    domains = data['dynamics'][0]['domains']
    steps = [r['step'] for r in data['dynamics'][0]['history']]
    for ax, method in zip(axes.flat, dynamic_methods):
        matrix = np.mean([[[s['methods'][method]['positive_composition'][d] for s in run['history']]
                           for d in domains] for run in data['dynamics']], axis=0)
        ax.stackplot(steps, matrix, labels=domains, colors=plt.get_cmap('tab20').colors[:len(domains)])
        ax.set_title(method)
        ax.set(xlabel='Actual optimizer updates', ylabel='Positive domain contribution (%)', ylim=(0,100))
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
    fig.tight_layout(rect=(0,0.14,1,1))
    for suffix in ('png','pdf','svg'):
        fig.savefig(out / f'domain_dynamics.{suffix}', dpi=200)
    plt.close(fig)
    source_paths = [BASE / 'inputs/dataset.json', *BASE.glob('*.py'),
                    BASE.parent / 'pile_case_study_common.py',
                    BASE.parent / 'table2_pile_memorization/corpora.py',
                    *out.glob('*/seed*/result.json'), *out.glob('*/seed*/scores.npz')]
    (out / 'source_manifest.json').write_text(json.dumps({str(p.relative_to(BASE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in source_paths}, indent=2) + '\n')
    print(f'Validated {len(rank_rows)} query ranks and {len(history_rows)} domain records; saved {out}')

if __name__ == '__main__':
    main()
