from retrieval.bootstrap import align_scores, paired_bootstrap, cluster_bootstrap, bonferroni_level, \
    min_detectable_effect, interpret
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SIZES = [64, 96, 128, 192, 254]
MARGIN = {'ndcg': 0.01, 'er@256': 0.02, 'er@512': 0.02, 'er@1024': 0.02}
PRIMARY_LEVEL = bonferroni_level(3)


def load_runs():
    
    runs = {}
    
    for path in sorted((RESULTS / "runs").glob("*.json")):
        
        run = json.loads(path.read_text(encoding='utf-8'))
        runs[run['name']] = run
    
    return runs

def compare(a_run, b_run, metric, level, clusters=None, label=""):
    
    qids, a, b = align_scores(a_run['per_query'], b_run['per_query'], metric)
    
    if clusters is not None:
        res = cluster_bootstrap(a, b, [clusters[q] for q in qids], level=level)
    else:
        res = paired_bootstrap(a, b, level=level)
    
    res['a'] = a_run['name']
    res['b'] = b_run['name']
    res['metric'] = metric
    res['label'] = label
    res['mean_a'] = float(a.mean())
    res['mean_b'] = float(b.mean())
    res['mde'] = float(min_detectable_effect(b - a, level))
    res['reading'] = interpret(res['ci_low'], res['ci_high'], MARGIN[metric])
    
    if clusters is not None:
        res['mde_note'] = 'question-level MDE; optimistic for clustered data (the cluster CI is the authority)'
    
    return res

def size_rule(runs, prefix, metric, clusters=None):
    
    means = {s: runs[f'{prefix}{s}']['scores'][metric] for s in SIZES}
    best = max(SIZES, key=lambda s: means[s])
    
    comparisons = []
    tied = [best]
    
    for s in SIZES:
        
        if s == best:
            continue
        
        res = compare(runs[f'{prefix}{best}'], runs[f'{prefix}{s}'], metric, 0.95, clusters,
                      label=f'size {s} vs best size {best}')
        comparisons.append(res)
        
        if res['ci_low'] <= 0 <= res['ci_high']:
            tied.append(s)
    
    return {
        'metric': metric,
        'means': {str(s): means[s] for s in SIZES},
        'best_size': best,
        'tied_with_best': sorted(tied),
        'chosen_size': min(tied),
        'note': 'provisional until the floor audit passes (rule 5d)',
        'comparisons': comparisons,
    }


if __name__ == "__main__":
    
    runs = load_runs()
    clusters = json.loads((RESULTS / "qasper_test_clusters.json").read_text(encoding='utf-8'))
    
    out = {}
    
    out['size_scifact'] = size_rule(runs, 'A1_fixed_', 'ndcg')
    out['size_qasper'] = size_rule(runs, 'B1_fixed_', 'er@512', clusters)
    
    fixed_q = runs['B1_fixed_128']
    primary = []
    
    for strategy in ['sentence', 'recursive', 'semantic']:
        primary.append(compare(fixed_q, runs[f'B2_{strategy}_128'], 'er@512', PRIMARY_LEVEL, clusters,
                               label=f'PRIMARY: {strategy} vs fixed @128'))
    
    out['primary'] = {'level': PRIMARY_LEVEL, 'comparisons': primary}
    
    exploratory = []
    fixed_s = runs['A1_fixed_128']
    
    for strategy in ['sentence', 'semantic']:
        exploratory.append(compare(fixed_s, runs[f'A2_{strategy}_128'], 'ndcg', 0.95,
                                   label=f'scifact: {strategy} vs fixed @128'))
    
    best_sci = out['size_scifact']['best_size']
    exploratory.append(compare(runs['A0_whole'], runs[f'A1_fixed_{best_sci}'], 'ndcg', 0.95,
                               label=f'scifact: fixed @{best_sci} vs whole document'))
    
    for metric in ['er@256', 'er@1024']:
        for strategy in ['sentence', 'recursive', 'semantic']:
            exploratory.append(compare(fixed_q, runs[f'B2_{strategy}_128'], metric, 0.95, clusters,
                                       label=f'qasper {metric}: {strategy} vs fixed @128'))
    
    out['exploratory'] = exploratory
    
    winners = [c for c in primary if c['reading'] == 'b_better']
    strategy = max(winners, key=lambda c: c['mean_diff'])['b'].split('_')[1] if winners else 'fixed'
    size = out['size_qasper']['chosen_size']
    out['stage2'] = {'strategy': strategy, 'size': size,
                     'command': f'python -m experiments.run_sweep stage2 {strategy} {size}'}
    
    (RESULTS / "comparisons").mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparisons" / "comparisons.json").write_text(json.dumps(out, indent=2), encoding='utf-8')
    
    for key in ['size_scifact', 'size_qasper']:
        r = out[key]
        print(f"\n{key}: best {r['best_size']}, tied with best {r['tied_with_best']}, chosen {r['chosen_size']}")
        for c in r['comparisons']:
            print(f"  {c['label']:<32} diff {c['mean_diff']:+.4f}  CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  {c['reading']}")
    
    print(f"\nprimary (level {PRIMARY_LEVEL:.4f}):")
    for c in primary:
        print(f"  {c['label']:<36} diff {c['mean_diff']:+.4f}  CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  "
              f"{c['reading']}  W/T/L {c['wins']}/{c['ties']}/{c['losses']}  MDE {c['mde']:.4f}")
    
    print("\nexploratory:")
    for c in exploratory:
        print(f"  {c['label']:<44} diff {c['mean_diff']:+.4f}  CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  {c['reading']}")
    
    print(f"\nstage 2: {out['stage2']['command']}")