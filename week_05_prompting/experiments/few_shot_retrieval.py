from prompting.llm import AnthropicClient
from prompting.classify import classify_intent
from pathlib import Path
from experiments.load_data import load_banking77
import pandas as pd
import json
from experiments.plots import plot_confusion_matrix, top_confusions
from jinja2 import Environment, FileSystemLoader
from prompting.sampler.sample_generator import top_similarity, capped_label
from prompting.sampler.embeddings import embed_queries

from experiments.constants import TEMPLATE_DIR, RESULTS_DIR, ARTIFACT_DIR, LABEL_COLUMN, QUERY_COLUMN
llm = AnthropicClient()

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)

system_template = env.get_template("system/classify_system.j2")
user_template = env.get_template("user/classify_few_shot.j2")

RESULTS_DIR.mkdir(exist_ok=True)


MODEL_NAME = "all-MiniLM-L6-v2"
DATA_NAME = "banking77"
SEED = 42

def run_few_shot():
    
    train, test = load_banking77(testing_size=50, seed=SEED)
    E, model = embed_queries(train, DATA_NAME, QUERY_COLUMN, MODEL_NAME)
    
    labels = train[LABEL_COLUMN].unique().tolist()
    
    system_prompt = system_template.render(categories=labels)
    
    sims_results = []
    capped_results = []
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        query = row[QUERY_COLUMN]
        label = row[LABEL_COLUMN]
        
        data, sel_emb_1, query_emb_1 = top_similarity(query, train, model, E, 5)
        
        query_answers = []
        
        for ex in data.itertuples():
            query_answers.append((ex.text, ex.label_name))
        
        user_prompt_sims = user_template.render(examples = query_answers, query = query)
        
        data2, sel_emb_2 , query_emb_2 = capped_label(query, train, model, E, LABEL_COLUMN, 0.5, 3)
        query_answers_2 = []
        for row in data2.itertuples():
            query_answers_2.append((row.text, row.label_name))

        user_prompt_capped = user_template.render(examples = query_answers_2, query = query)

        out1 = classify_intent(llm, system_prompt=system_prompt, user_prompt=user_prompt_sims, valid_labels= set(labels))
        out2 = classify_intent(llm, system_prompt=system_prompt, user_prompt=user_prompt_capped, valid_labels= set(labels))

        sims_results.append(format_response(idx, query, label, data, out1, sel_emb_1, query_emb_1))
        capped_results.append(format_response(idx, query, label, data2, out2, sel_emb_2, query_emb_2))
    
    
    df_sims = pd.DataFrame(sims_results)
    df_capped = pd.DataFrame(capped_results)
    
    df_sims.to_csv(RESULTS_DIR / "few_shot_similarity_retrieval.csv", index=False, encoding='utf-8')
    df_capped.to_csv(RESULTS_DIR / "few_shot_capped_retrieval.csv", index=False, encoding='utf-8')
    
    if len(df_sims.loc[df_sims['correct'] == False]) > 0:
        df_sims_errors = df_sims.loc[df_sims['correct'] == False]
        df_sims_errors.to_csv(RESULTS_DIR / "few_shot_similarity_retrieval_errors.csv", index=False, encoding='utf-8')
    
    if len(df_capped.loc[df_capped['correct'] == False]) > 0:
            df_capped_errors = df_capped.loc[df_capped['correct'] == False]
            df_capped_errors.to_csv(RESULTS_DIR / "few_shot_capped_retrieval_errors.csv", index=False, encoding='utf-8')

    sims_summary = aggregate_summary(df_sims, len(df_sims), 'Top_Similarity_Retrieval', SEED)
    
    (RESULTS_DIR / "few_shot_top_similarity.json").write_text(json.dumps(sims_summary, indent= 2), encoding='utf-8')
    
    capped_summary = aggregate_summary(df_capped, len(df_capped), 'Capped_Similarity_Retrieval', SEED)
    
    (RESULTS_DIR / "few_shot_capped_similarity.json").write_text(json.dumps(capped_summary, indent= 2), encoding='utf-8')

def aggregate_summary(data, n, method, seed=SEED):
    
    total_correct = len(data.loc[data['correct']])
    
    accuracy = float( total_correct / len(data))
    invalid_count = len(data.loc[data['valid'] == False])
    total_cost = float(data['cost'].sum())
    
    cost_per_correct = float(total_cost / total_correct) if total_correct > 0 else None
    retrieval_hit_rate = float(data['hit'].sum() / len(data))
    top_1_avg_score = float(data['top_1_score'].mean())
    pred_top1_agreemnet_rate = float(len(data.loc[data['pred_equals_top1']]) / len(data))
     
    summary = {'method': method, 'n': n, 'seed': seed, 'accuracy': accuracy, 'invalid_count': invalid_count, 'total_cost': total_cost, 'cost_per_correct': cost_per_correct, 'retrieval_hit_rate': retrieval_hit_rate, 'top_1_avg_score': top_1_avg_score, 'pred_top1_agreement_rate': pred_top1_agreemnet_rate}
    
    return summary

def format_response(idx, query, label, samp_data, output, sel_emb, query_emb):
    
    pred, valid, response = output.label, output.valid, output.raw
    
    is_correct = label == pred if valid else False
    sample_labels = "|".join(samp_data[LABEL_COLUMN].tolist())
    correct_labels = len(samp_data.loc[samp_data[LABEL_COLUMN] == label])
    hit = 1 if correct_labels > 0 else 0
    top_1_label = samp_data.iloc[-1][LABEL_COLUMN]
    #print(sel_emb[-1].dot(query_emb))
    
    pred_equals_top1 = samp_data.iloc[-1][LABEL_COLUMN] == pred if valid else False
    top_1_score = sel_emb[-1].dot(query_emb)
    return {'row_id': idx, 'text': query, 'true_label': label, 'pred_label': pred, 'valid': valid, 'correct': is_correct, 'stop_reason': response.stop_reason, 'input_tokens': response.input_tokens, 'output_tokens': response.output_tokens, 'cost': response.cost, "example_labels": sample_labels, 'hit': hit,
            'top_1_label': top_1_label, 'top_1_score': top_1_score, 'pred_equals_top1': pred_equals_top1}
    
if __name__ == "__main__":
    run_few_shot()
            
    