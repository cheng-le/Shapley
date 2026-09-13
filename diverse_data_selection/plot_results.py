"""Plot paired module ablations from the completed summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    rows = json.loads((args.results / 'summary.json').read_text())
    styles = [('Adam Top-k', '#555555', 'o'), ('Graph Top-k', '#2878b5', 's'),
              ('Adam + diversity', '#bd5c25', '^'), ('Graph + diversity', '#16816a', 'D')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for column, order in enumerate((1, 2)):
        for row, (metric, label) in enumerate((('val_loss', 'Validation loss'),
                                               ('affinity_per_item', 'Graph affinity per selected item'))):
            axis = axes[row, column]
            for arm, color, marker in styles:
                group = sorted([x for x in rows if x['order'] == order and x['arm'] == arm], key=lambda x: x['budget'])
                if any(x[metric] is None for x in group):
                    raise ValueError('Complete all utility cells before plotting final results')
                axis.errorbar([x['budget'] for x in group], [x[metric]['mean'] for x in group],
                              yerr=[x[metric]['std'] for x in group], label=arm,
                              color=color, marker=marker, markersize=4, capsize=2)
            axis.set_title(f'Order {order}')
            axis.set_xlabel('Retained training documents (%)')
            axis.set_ylabel(label)
            axis.set_xticks(sorted({x['budget'] for x in rows}))
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(args.results / f'module_ablation.{suffix}', dpi=220)
    plt.close(fig)

if __name__ == '__main__':
    main()
