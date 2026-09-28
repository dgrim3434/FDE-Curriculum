from prompting.llm import AnthropicClient
from prompting.extraction import extraction
from prompting.budget import BudgetExceeded
from jinja2 import Environment, FileSystemLoader
from experiments.load_data import load_banking77
from experiments.constants import TEMPLATE_DIR, RESULTS_DIR, QUERY_COLUMN, LABEL_COLUMN
import re
import pandas as pd
import json
from prompting.extraction import get_schema_template

llm = AnthropicClient()
SEED = 42

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=0)
system_template = env.get_template("system/extraction.j2")
user_template = env.get_template("user/structured_json.j2")

def run_extraction():
    
    train, test = load_banking77(testing_size=50, seed=SEED)
    
    labels = train[LABEL_COLUMN].unique().tolist()
    
    budget = BudgetExceeded(1.00)
    
    schema = get_extraction_schema(labels)
    
    schema_template = get_schema_template(schema)
    
    system_prompt = system_template.render(fields = schema_template)
    
    results = []
    
    for idx in range(len(test)):
        
        row = test.iloc[idx]
        
        label = row[LABEL_COLUMN]
        query = row[QUERY_COLUMN]
        
        user_prompt = user_template.render(query = query)
        resp = extraction(llm, system_prompt,user_prompt, schema, budget, schema_template)
        
        
        result = {'idx': idx, 'query': query, 'true_label': label, 'ok': resp.ok, 'reason': resp.reason, 'attempts': resp.attempts, 'cost': resp.total_cost, 'error_log': "| ".join(resp.error_log)}
        
        if resp.ok:
            pred = resp.data['topic']
            result['pred_topic'] = pred
            result['topic_correct'] = pred == label
            result['urgency'] = resp.data['urgency']
            
            amount = resp.data['amount_mentioned']
            result['amount_mentioned'] = amount
            result['amount_check'] = search_extracted_amount(query, amount)

        else:
            result['pred_topic'] = None
            result['topic_correct'] = False
            result['urgency'] = None
            result['amount_mentioned'] = None
            result['amount_check'] = None

        results.append(result)
    
    df = pd.DataFrame(results)

    df.to_csv(RESULTS_DIR / "extraction/extraction.csv", index=False, encoding='utf-8')
    
    correct_topic = len(df.loc[df['topic_correct']])
    accuracy = float(correct_topic / len(df))
    avg_attempts = float(df['attempts'].mean())
    # ["low", "medium", "high"]
    high_pct = float(len(df.loc[df['urgency'] == 'high']) / len(df))
    med_pct = float(len(df.loc[df['urgency'] == 'medium']) / len(df))
    low_pct = float(len(df.loc[df['urgency'] == 'low']) / len(df))
    total_cost = float(df['cost'].sum())
    correct_amount = len(df.loc[(df['amount_check'] == 'none_ok') | (df['amount_check'] == 'grounded')])
    
    
    summary = {'method': 'extraction', 'n': len(df), 'seed': SEED, 'accuracy': accuracy, 'average_attempts': avg_attempts, 'high_urgency_pct': high_pct, 'low_urgency_pct': low_pct, 'medium_urgency_pct': med_pct,
               'total_cost': total_cost, 'correct_extracted_amount': correct_amount}
    
    (RESULTS_DIR / "extraction/extraction.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
        

def search_extracted_amount(query, amount_mentioned):
    
    if amount_mentioned is not None:
        
        nums = re.findall(r"\d[\d,]*(?:\.\d+)?", query)
        
        res = []
        if len(nums) > 0:
            for val in nums:
                res.append(float(re.sub(r"[£$€,\s]", "", val)))
            
            for num in res:
                if abs(num - amount_mentioned) < 0.01:
                    return "grounded"
            
        return "not_in_text"
    
    else:
        check_1 = re.findall(r"[£$€]\s?\d[\d,]*(?:\.\d+)?", query)
        check_2 = re.findall(r"\d[\d,]*(?:\.\d+)?\s?(?:pounds?|dollars?|euros?|gbp|usd|eur)\b", query, flags=re.IGNORECASE)
        
        if len(check_1) > 0 or len(check_2) > 0:
            return "missed"
    
    return "none_ok"

def get_extraction_schema(label_names: str) -> dict:
    
    EXTRACTION_SCHEMA = {
        "topic": {
            "type": (str,),
            "nullable": False,
            "allowed": label_names,
            "description": "The single intent that best describes what the customer is asking for. All valid valid intents all listed above.",
        },
        "urgency": {
            "type": (str,), "nullable": False, "allowed": ["low", "medium", "high"],
            "description": (
                "high = possible fraud; lost, stolen or compromised card; or money missing. "
                "medium = money stuck, pending, failed or declined, or the customer is blocked from using their account. "
                "low = general questions, how-to, fees, limits, or information requests."
            ),
        },
        "amount_mentioned": {
            "type": (int, float), "nullable": True,
            "description": (
                "The money amount written in the message as a plain number, no currency symbol or commas "
                "(e.g. $1,250.50 -> 1250.5). Use null if no amount is mentioned. "
                "If several amounts appear, use the one the request is about."
            ),
        }
    }
    
    return EXTRACTION_SCHEMA


if __name__ == "__main__":
    run_extraction()