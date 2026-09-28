from prompting.sampler.embeddings import embed_queries
from prompting.sampler.sample_generator import random_sampler, top_similarity, top_diverse, capped_label
from experiments.load_data import load_banking77
import numpy as np
import pandas as pd
from pathlib import Path
from experiments.constants import RESULTS_DIR, LABEL_COLUMN, QUERY_COLUMN


MODEL_NAME = "all-MiniLM-L6-v2"
DATA_NAME = "banking77"
SEED = 42

# return data.iloc[results], E[idx], embedding

def run_experiment(k, results, train, test, E, model):
    
    
    
    # Logging results
    random_results = []
    top_sims = []
    top_div = []
    capped_lab = []
    
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        
        query = row[QUERY_COLUMN]
        
        data, sel_emb, query_emb = random_sampler(query, train, model, E, k, SEED)
        random_results.append(analyze_results(row, data, sel_emb, query_emb))
        
        data, sel_emb, query_emb = top_similarity(query, train, model, E, k)
        top_sims.append(analyze_results(row, data, sel_emb, query_emb))
        
        data, sel_emb, query_emb = top_diverse(query, train, model, E, 0.7, k)
        top_div.append(analyze_results(row, data, sel_emb, query_emb))
        
        data, sel_emb, query_emb = capped_label(query, train, model, E, LABEL_COLUMN, 0.5, k)
        capped_lab.append(analyze_results(row, data, sel_emb, query_emb))
        
    results.append(convert_to_mean_dict('Random', k, random_results))
    results.append(convert_to_mean_dict('Top Similarity', k, top_sims))
    results.append(convert_to_mean_dict('Top Diverse', k, top_div))
    results.append(convert_to_mean_dict('Capped Label', k, capped_lab))

    return results
    
        
def convert_to_mean_dict(method, samples, data):
    df = pd.DataFrame(data)
    
    result = {'method': method, 'samples': samples}
    
    for col in df.columns:
        result[col] = float(df[col].mean())
    
    return result
    
    

def analyze_results(query_row, results, result_emb, query_emb):
    
    label =  query_row[LABEL_COLUMN]
    
    correct_labels = len(results.loc[results[LABEL_COLUMN] == label])
    
    hit = 1 if correct_labels > 0 else 0
    top_hit = 1 if results.iloc[-1][LABEL_COLUMN] == label else 0
    precision = float(correct_labels / len(results))
    distinct_labels = results[LABEL_COLUMN].nunique()
    
    
    mean_score = (result_emb @ query_emb).mean()
    
    mask = np.eye(result_emb.shape[0], dtype=bool)
    
    redundancy = (result_emb @ result_emb.T)[~mask].mean()
    
    return {'hit': hit, 'top_hit': top_hit, 'precision': precision, 'distinct_labels': distinct_labels, 'mean_score': mean_score, 'redundancy': redundancy}
    
def run():
    
    ks = [1,3,5]
    
    train, test = load_banking77(testing_size=50, seed=SEED)
        
    E, model = embed_queries(train, DATA_NAME, "text", MODEL_NAME)
    results = []
    
    # run_experiment(k, results, train, test, E, model):
    for k in ks:
        
        results = run_experiment(k, results, train, test, E, model)
    
    df = pd.DataFrame(results)
    
    df.to_csv(RESULTS_DIR / "sampler_comparison.csv", index= False, encoding='utf-8')
    
    print(df.head(len(df)))

if __name__ == "__main__":
    run()
    