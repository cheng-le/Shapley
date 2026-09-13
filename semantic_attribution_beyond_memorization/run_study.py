"""Record a true local training trajectory for semantic case-study comparisons."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from pile_case_study_common import (LocalPileProtocol, batch_gradients, bm25_scores,
    build_semantic_graph, graph_regularize, learning_rate_at, load_attribution_model,
    load_tokenizer, make_optimizer, semantic_embeddings, set_seed, tokenize_texts, update_model)

BASE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['paraphrase', 'dynamics'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    path = BASE / 'inputs/dataset.json'
    data = json.loads(path.read_text())
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    out = BASE / 'results' / args.task / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    protocol = LocalPileProtocol(batch_size=8, seed=args.seed)
    set_seed(args.seed)
    is_paraphrase = args.task == 'paraphrase'
    records = data['paraphrase_train' if is_paraphrase else 'dynamics_train']
    queries = data['queries' if is_paraphrase else 'math_validation']
    texts = [r['text'] for r in records]
    query_texts = [r['text'] for r in queries]
    domains = np.asarray([r['domain'] for r in records])
    tokenizer = load_tokenizer('gpt2')
    model = load_attribution_model(protocol)
    optimizer = make_optimizer(model, protocol)
    optimizer.defaults['foreach'] = False
    for group in optimizer.param_groups:
        group['foreach'] = False
    embeddings = semantic_embeddings(model, tokenizer, texts, protocol.sequence_length)
    query_embeddings = semantic_embeddings(model, tokenizer, query_texts, protocol.sequence_length)
    neighbors, weights = build_semantic_graph(embeddings, protocol.graph_k)
    tokens = tokenize_texts(tokenizer, texts, protocol.sequence_length)
    query_tokens = tokenize_texts(tokenizer, query_texts, protocol.sequence_length)
    query_count = len(queries) if is_paraphrase else 1
    first = np.zeros((query_count, len(records)))
    second = np.zeros_like(first)
    seen = np.zeros(len(records), dtype=bool)
    visits = np.zeros(len(records), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    steps = int(np.ceil(len(records) / protocol.batch_size)) if is_paraphrase else 500
    order, cursor = rng.permutation(len(records)), 0
    history, losses = [], []
    for step in range(steps):
        if cursor == len(order):
            order, cursor = rng.permutation(len(records)), 0
        indices = order[cursor:cursor + protocol.batch_size]
        cursor += len(indices)
        batch = {k: v[indices] for k, v in tokens.items()}
        train_grad, _ = batch_gradients(model, batch)
        query_grad, _ = batch_gradients(model, query_tokens)
        if not is_paraphrase:
            query_grad = query_grad.mean(0, keepdim=True)
        lr = learning_rate_at(step, steps, protocol)
        local_first = lr / len(indices) * (query_grad @ train_grad.T)
        curvature = 0.5 * lr * lr / len(indices) * ((query_grad.square() * train_grad.mean(0)) @ train_grad.T)
        curvature *= torch.clamp(0.25 * local_first.std(unbiased=False).clamp_min(1e-12) /
                                 curvature.std(unbiased=False).clamp_min(1e-12), max=1.0)
        first[:, indices] += local_first.detach().cpu().numpy()
        second[:, indices] += (local_first - curvature).detach().cpu().numpy()
        seen[indices] = True
        visits[indices] += 1
        loss = update_model(model, optimizer, batch, lr, protocol.grad_clip)
        losses.append(loss)
        if not is_paraphrase and (step == 0 or (step + 1) % 10 == 0):
            snapshot = {'step': step + 1, 'seen_count': int(seen.sum()), 'methods': {}}
            for name, values in [('In-Run 1st (local)', first), ('In-Run 2nd (local)', second)]:
                graph = graph_regularize(values, neighbors, weights, protocol.graph_alpha, protocol.graph_lambda)
                for method, scores in [(name, values[0]), ('Graph ' + name, graph[0])]:
                    totals, averages = {}, {}
                    for domain in sorted(set(domains)):
                        mask = (domains == domain) & seen
                        totals[domain] = float(scores[mask].sum())
                        averages[domain] = float(scores[mask].mean()) if mask.any() else None
                    positive_mass = sum(max(x, 0) for x in totals.values())
                    composition = {d: 100 * max(v, 0) / positive_mass if positive_mass else 0 for d, v in totals.items()}
                    snapshot['methods'][method] = {'signed_totals': totals, 'seen_item_means': averages,
                                                   'positive_composition': composition}
            history.append(snapshot)
        if (step + 1) % 25 == 0 or step + 1 == steps:
            print(f'{args.task} seed={args.seed} step={step+1}/{steps} loss={loss:.4f}', flush=True)
    graph_first = graph_regularize(first, neighbors, weights, protocol.graph_alpha, protocol.graph_lambda)
    graph_second = graph_regularize(second, neighbors, weights, protocol.graph_alpha, protocol.graph_lambda)
    scores = {'In-Run 1st (local)': first, 'In-Run 2nd (local)': second,
              'Graph In-Run 1st (local)': graph_first, 'Graph In-Run 2nd (local)': graph_second}
    ranks = []
    if is_paraphrase:
        # Use the same visible text window for lexical and neural retrieval.
        visible_texts = [tokenizer.decode(ids[mask.bool()].tolist()) for ids, mask in
                         zip(tokens['input_ids'], tokens['attention_mask'])]
        visible_queries = [tokenizer.decode(ids[mask.bool()].tolist()) for ids, mask in
                           zip(query_tokens['input_ids'], query_tokens['attention_mask'])]
        scores['BM25 (local)'] = bm25_scores(visible_texts, visible_queries)
        scores['Embedding cosine'] = query_embeddings @ embeddings.T
        for method, values in scores.items():
            for index, query in enumerate(queries):
                rank = float(rankdata(-values[index], method='average')[query['target']])
                ranks.append({'method': method, 'query': index, 'category': query['category'],
                              'group': query['group'], 'target': query['target'], 'rank': rank,
                              'reciprocal_rank': 1 / rank, 'hit_at_10': int(rank <= 10)})
    payload = {'task': args.task, 'seed': args.seed, 'data_sha256': data_hash,
               'protocol': asdict(protocol), 'train_count': len(records), 'steps': steps,
               'approximation': 'Final-LayerNorm SGD-proxy scores along AdamW training; second order is capped diagonal-Fisher surrogate. Graph is final-score regularization, not the online draft algorithm.',
               'ranks': ranks, 'history': history, 'losses': losses,
               'visits': visits.tolist(), 'domains': sorted(set(domains))}
    np.savez_compressed(out / 'scores.npz', **scores, neighbors=neighbors, weights=weights)
    (out / 'result.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
