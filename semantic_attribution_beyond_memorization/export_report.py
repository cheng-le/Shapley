"""Report the requested legacy ranking table and actual domain-study examples."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from pile_case_study_common import LocalPileProtocol, load_attribution_model, load_tokenizer, semantic_embeddings, set_seed

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

def main():
    legacy_path = ROOT / 'table2_pile_memorization/results/table2_results.json'
    legacy = json.loads(legacy_path.read_text())
    input_path = BASE / 'inputs/dataset.json'
    data = json.loads(input_path.read_text())
    train = data['dynamics_train']
    validation = data['math_validation']
    scores = np.stack([np.load(BASE / f'results/dynamics/seed{s}/scores.npz')['Graph In-Run 2nd (local)'][0] for s in range(3)])
    means = scores.mean(0)
    set_seed(0)
    protocol = LocalPileProtocol()
    tokenizer = load_tokenizer('gpt2')
    model = load_attribution_model(protocol)
    embeddings = semantic_embeddings(model, tokenizer, [r['text'] for r in validation], 128)
    centrality = ((embeddings @ embeddings.T).sum(1) - 1) / (len(validation) - 1)
    medoid = int(np.argmax(centrality))
    examples = []
    for label, index, records in [('highest', int(means.argmax()), train),
                                  ('lowest', int(means.argmin()), train),
                                  ('validation_medoid', medoid, validation)]:
        record = records[index]
        ids = tokenizer(record['text'], max_length=128, truncation=True)['input_ids']
        item = {'kind': label, 'index_zero_based': index, 'domain': record['domain'],
                'sha256': record['sha256'], 'full_text': record['text'],
                'visible_text_128_tokens': tokenizer.decode(ids), 'visible_token_count': len(ids)}
        if records is train:
            item.update(mean_score=float(means[index]), seed_scores=scores[:, index].tolist())
        else:
            item['mean_cosine_to_other_validation_samples'] = float(centrality[index])
        examples.append(item)
    counts = dict(sorted(Counter(r['domain'] for r in train).items()))
    output = {'legacy_ranking_source': str(legacy_path.relative_to(ROOT)),
              'legacy_ranking_corpus_size': legacy['actual']['corpus_count'],
              'legacy_ranking_seed': legacy['protocol']['local_executable']['seed'],
              'domain_counts': counts, 'train_total': len(train), 'validation_total': len(validation),
              'extrema_rule': 'Highest/lowest mean final Graph second-order local score across seeds 0/1/2',
              'validation_rule': 'Maximum mean cosine to the other 15 validation samples using frozen pretrained GPT-2, first 128 tokens',
              'examples': examples, 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()}
    out = BASE / 'results'
    (out / 'report_examples.json').write_text(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    categories = list(next(iter(legacy['summary'].values()))['average_rank'])
    lines = ['# Requested Reporting Version', '', '## Paraphrase Ranking (Legacy Fusion Case Study)', '',
             'This table uses 512 candidates and seed 0. Each entry averages the designated-source ranks over three query groups. It is not a three-seed result or the newer 2,048-candidate graph-only experiment.', '',
             '| Method | ' + ' | '.join(categories) + ' |', '|---|' + '---:|' * 4]
    for method, result in legacy['summary'].items():
        lines.append('| ' + method + ' | ' + ' | '.join(f'{result["average_rank"][c]:.1f}' for c in categories) + ' |')
    lines.extend(['', 'Suggested reporting text:', '',
                  'In the 512-candidate local case study, the graph-based fusion variants achieve mean source ranks of 1.0, 1.0, 1.3, and 26.7 for partial-exact, paraphrased, significantly paraphrased, and similar-topic queries, respectively. They tie for the best ranks in the first two categories and obtain the best rank under significant paraphrasing. For similar-topic queries, their rank of 26.7 is close to the local influence-function baseline (26.0) and better than BM25 (231.3). The graph rows combine semantic and lexical retrieval with graph-regularized in-run scores; these results evaluate that fusion system and do not isolate the effect of graph propagation.', '',
                  'The newer graph-only result remains in summary.md. The two settings must not be merged as one controlled experiment.', '',
                  '## Latest Domain-Dynamics Figure', '',
                  '1,817 training documents, 17 domains, 16 separate math validation documents; seeds 0/1/2 and 500 updates per seed. Training inputs are truncated to 128 tokens.', '',
                  '| Domain | Training documents |', '|---|---:|'])
    for domain, count in counts.items():
        lines.append(f'| {domain} | {count} |')
    lines.extend(['', 'The legacy figure3_pile_graph_dynamics result used 32 documents per domain (544 total) and four math validation documents. It is a separate dataset snapshot and protocol.', '',
                  '## Reproducibly Selected Examples', '',
                  'Highest/lowest refer to the mean final Graph second-order local score over seeds 0/1/2. They are extrema, not estimates of the typical document in each domain. The validation exemplar maximizes mean cosine to the other validation documents using frozen initial GPT-2 representations.', ''])
    for item in examples:
        lines.extend([f'### {item["kind"]}: {item["domain"]}, index {item["index_zero_based"]}', '',
                      f'SHA256: `{item["sha256"]}`', ''])
        if 'mean_score' in item:
            lines.extend([f'Mean final score: {item["mean_score"]:+.6e}.', ''])
        lines.extend(['Visible input window:', '', '```text', item['visible_text_128_tokens'], '```', ''])
    lines.append('A low score is relative to the mathematical utility objective; it does not imply that the original document is generally low quality.')
    (out / 'requested_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in examples[-1].items() if k != 'full_text'}, indent=2))

if __name__ == '__main__':
    main()
