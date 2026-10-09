import os, sys, json, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = str(args.threads)

import numpy as np
from threadpoolctl import threadpool_info
from search.brute_force import BruteForceIndex, normalize, top_k_batch
from search.ivf import IVFIndex
from search.metrics import index_recall
from pathlib import Path

EMB_PATH = Path(__file__).resolve().parent.parent / "data/embeddings"
RESULTS = Path(__file__).resolve().parent.parent / "results"

def time_queries(search_fn, Q, warmup=20):
    
    for q in Q[:warmup]:
        search_fn(q)
    
    times = []
    
    for q in Q:
        t0 = time.perf_counter()
        search_fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    
    return np.array(times)

SEED = 0
N_PROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 723]
Lat_SWEEP = [1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 522_931]


if __name__ == "__main__":
    rng = np.random.default_rng(seed=SEED)
    corpus = np.load(EMB_PATH / "quora_minilm_corpus.npy")
    queries = np.load(EMB_PATH / "quora_minilm_queries.npy")
    
    
    bf = BruteForceIndex(corpus)
    
    nlist = int(np.sqrt(len(corpus)))
    
    train_size = nlist * 256
    t0 = time.perf_counter()
    ivf = IVFIndex(corpus, nlist, seed=SEED, train_size=train_size)
    t_build = time.perf_counter() - t0
    
    indicies = rng.choice(len(queries), size=1000, replace=False)
    
    test_queries = queries[indicies]
    
    br_indices = bf.search_batch(test_queries, k=100)[0]
    
    br_times = time_queries(lambda q: bf.search(q, k=10), test_queries)
    p50, p95 = np.percentile(br_times, [50, 95])
    results = {'brute_force': {'p50_latency': p50, 'p95_latency': p95, 'recall': '1.0, reference'}}
    
    clusters = ivf.cluster_sizes()
    results['ivf_info'] = {'build_time': t_build, 'min_cluster_size': int(clusters.min()), 'median_cluster_size': int(np.median(clusters)), 'max_cluster_size': int(clusters.max())}
    
    results['experiment'] = []
    
    CONFIG = {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'd': len(corpus[0]), 'seed': SEED, 'warm_up': 20, 'latency_k': 10,
              'recall_k': 100, 'threads': str(args.threads), 'thread_info': threadpool_info(), 'numpy_version': np.__version__}
    
    result_config = CONFIG.copy()
    result_config['number_of_queries'] = 1000
    result_config['N'] = len(corpus)
    result_config['nlist'] = nlist
    result_config['train_size'] = train_size
    results['config'] = result_config
    
    
    for n_probe in N_PROBES:
        
        probe_times = time_queries(lambda q: ivf.search(q, k=10, nprobe=n_probe), test_queries)
        p50, p95 = np.percentile(probe_times, [50, 95])
        
        probe_index = ivf.search_batch(test_queries, k=100, nprobe=n_probe)[0]
        
        result = {'nprobe': n_probe, 'p50_latency': p50, 'p95_latency': p95, 'index_recall@1': index_recall(probe_index, br_indices, k=1)[0], 'index_recall@10': index_recall(probe_index, br_indices, k=10)[0],
                  'index_recall@100': index_recall(probe_index, br_indices, k=100)[0]}
        
        # q (B, D) : centroids: (c, D) need (B, C): then take top n_probes. Then pass them into the number of values there
        
        scores = normalize(test_queries) @ ivf.centroids.T
        selected = top_k_batch(scores, n_probe)[0]
        
        sizes = ivf.cluster_sizes()[selected].sum(axis=1)
        avg_size = sizes.mean()
        
        result['perc_corpus_scanned'] = float(avg_size / ivf.N) * 100

        results['experiment'].append(result)
    
    
    lat_config = CONFIG.copy()
    lat_config['number_of_queries'] = 500
    lat_config['nprobe'] = 8
    lat_config['train_size'] = 'all'
    
    
    lat_results = {}
    lat_results['config'] = lat_config
    lat_results['experiment'] = []
    queries = test_queries[:500]
    
    for N in Lat_SWEEP:
        
        indicies = rng.choice(len(corpus), N, replace=False)
        corp = corpus[indicies]
        
        bf = BruteForceIndex(corp)
        
        n_list = int(np.sqrt(N))
        
        t0 = time.perf_counter()
        ivf = IVFIndex(corp, nlist=n_list, seed=SEED)
        build_time = time.perf_counter() - t0
        
        brute_func = lambda q: bf.search(q, k=10)
        brute_times = time_queries(brute_func, queries)
        
        ivf_func = lambda q: ivf.search(q, k=10, nprobe=8)
        ivf_time = time_queries(ivf_func, queries)
        
        b_50, b_95 = np.percentile(brute_times, [50, 95])
        i_50, i_95 = np.percentile(ivf_time, [50, 95])
        
        gold_index = bf.search_batch(queries, k=100)[0]
        ivf_index = ivf.search_batch(queries, k=100, nprobe=8)[0]
        
        result = {'N': N, 'brute_latency_p50': float(b_50), 'brute_latency_p95': float(b_95), 'ivf_latency_p50': float(i_50), 'ivf_latency_p95': float(i_95),
                  'index_recall@10': index_recall(ivf_index, gold_index, 10)[0], 'ivf_build_time': build_time, 'n_list': n_list}
        
        lat_results['experiment'].append(result)
    
    random_corpus = rng.normal(size=(len(corpus), len(corpus[0]))).astype(np.float32)
    
    bf_rand = BruteForceIndex(random_corpus)
    
    nlist = int(np.sqrt(len(random_corpus)))
    train_size = nlist * 256
    
    ivf_rand = IVFIndex(random_corpus, nlist=nlist, train_size=train_size)
    
    queries = rng.normal(size=(500, len(corpus[0]))).astype(np.float32)
    
    ground_rand = bf_rand.search_batch(queries, k=100)[0]

    rand_results = {}
    
    rand_results['experiments'] = []
    
    rand_config = CONFIG.copy()
    rand_config['number_of_queries'] = 500
    rand_config['nlist'] = nlist
    rand_config['train_size'] = train_size
    rand_config['N'] = len(random_corpus)
    
    rand_results['config'] = rand_config
    
    for n_probe in N_PROBES:
        
        probe_index = ivf_rand.search_batch(queries, k=100, nprobe=n_probe)[0]
        
        result = {'nprobe': n_probe, 'index_recall@1': index_recall(probe_index, ground_rand, k=1)[0], 'index_recall@10': index_recall(probe_index, ground_rand, k=10)[0],
                  'index_recall@100': index_recall(probe_index, ground_rand, k=100)[0]}
        
        # q (B, D) : centroids: (c, D) need (B, C): then take top n_probes. Then pass them into the number of values there
        
        scores = normalize(queries) @ ivf_rand.centroids.T
        selected = top_k_batch(scores, n_probe)[0]
        
        sizes = ivf_rand.cluster_sizes()[selected].sum(axis=1)
        avg_size = sizes.mean()
        
        result['perc_corpus_scanned'] = float(avg_size / ivf_rand.N) * 100

        rand_results['experiments'].append(result)
    
    
    threads = str(args.threads)
    
    (RESULTS / f'bench_nprobe_t{threads}.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (RESULTS / f'bench_scaling_t{threads}.json').write_text(json.dumps(lat_results, indent=2), encoding='utf-8')
    (RESULTS / f'rand_nprobe_t{threads}.json').write_text(json.dumps(rand_results, indent=2), encoding='utf-8')