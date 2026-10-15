import numpy as np
from statistics import NormalDist

def align_scores(per_query_a, per_query_b, metric) -> tuple[list, np.ndarray, np.ndarray]:
    
    queries_a = set([k for k,v in per_query_a.items()])
    queries_b = set([k for k, v in per_query_b.items()])
    
    if queries_a != queries_b:
        raise ValueError("ERROR: Results hold different sets of queries")
    
    qids = sorted(per_query_a)
    
    a = np.array([per_query_a[q][metric] for q in qids])
    b = np.array([per_query_b[q][metric] for q in qids])
    
    return qids, a, b

def paired_bootstrap(a, b, n_boot=10_000, seed=0, level = 0.95) -> dict:
    
    rng = np.random.default_rng(seed=seed)
    n = len(a)
    
    idx = rng.integers(0, n, size=(n_boot, n))
    
    d = b - a
    
    boot_means = d[idx].mean(axis=1)
    
    lo_pct = (1- level) / 2 * 100
    hi_pct = (1 + level) / 2 * 100
    
    lo, hi = np.percentile(boot_means, [lo_pct, hi_pct])
    
    results = {'mean_diff': float(d.mean()), 'ci_low': float(lo), 'ci_high': float(hi), 'level': level, 'n': n, 'n_boot': n_boot, 'seed': seed}
    
    comp = {'wins': 0, 'losses': 0, 'ties': 0}
    
    for i in range(n):
        
        if b[i] > a[i]:
            comp['wins'] += 1
        elif a[i] > b[i]:
            comp['losses'] += 1
        else:
            comp['ties'] += 1
    
    results.update(comp)
    
    return results

def cluster_bootstrap(a, b, clusters, n_boot=10_000, seed=0, level=0.95) -> dict:
    
    rng = np.random.default_rng(seed=seed)
    
    if len(clusters) == 0:
        raise ValueError("ERROR: length of clusters is 0")
    
    results = {}
    
    for i in range(len(clusters)):
        
        if clusters[i] not in results:
            
            results[clusters[i]] = {'sum': 0, 'count': 0}
        
        results[clusters[i]]['sum'] += b[i] - a[i]
        results[clusters[i]]['count'] += 1

        
    n_papers = len(results)
    
    qids = sorted([k for k,_ in results.items()])
    
    sums = np.array([results[i]['sum'] for i in qids])
    counts = np.array([results[i]['count'] for i in qids])
    #breakpoint()
    idx = rng.integers(0, n_papers, size=(n_boot, n_papers))
    boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    
    lo_pct = (1 - level) / 2 * 100
    hi_pct = (1 + level) / 2 * 100
    
    lo, hi = np.percentile(boot_means, [lo_pct, hi_pct])
    
    mean_diff = sums.sum() / counts.sum()
    results = {'mean_diff': float(mean_diff), 'ci_low': float(lo), 'ci_high': float(hi), 'level': level, 'n': len(a), 'n_clusters': n_papers, 'n_boot': n_boot, 'seed': seed}
    
    comp = {'wins': 0, 'losses': 0, 'ties': 0}
    
    for i in range(len(a)):
        
        if b[i] > a[i]:
            comp['wins'] += 1
        elif b[i] < a[i]:
            comp['losses'] += 1
        else:
            comp['ties'] += 1
    
    results.update(comp)
    
    return results

def bonferroni_level(m, alpha=0.05) -> float:
    
    return 1 - alpha / m

def min_detectable_effect(d, level=0.95) -> float:
    
    z = NormalDist().inv_cdf((1 + level) / 2)
    
    return z * np.std(d, ddof=1) / np.sqrt(len(d))

def interpret(ci_low, ci_high, tie_margin) -> str:
    
    
    if ci_low <= 0 and ci_high >= 0:
        
        if abs(ci_low) <= tie_margin and abs(ci_high) <= tie_margin:
            return 'tie'
        
        return 'inconclusive'
    
    if ci_low > 0:
        return 'b_better'
    
    return 'a_better'