from demos.hotpot_tools import make_hotpot_tools
from demos.load_data import load_hot_qa
from prompting.templates import render
from prompting.react import run_react
from prompting.llm import OllamaClient
from demos.constants import RESULTS_DIR
from prompting.parsing import parse_react, f1, exact_match
import json
import pandas as pd
import difflib


MODEL = "llama3.2:3b"
llm = OllamaClient(model=MODEL)

TOOL_DESCRIPTIONS = {
    "search":    "give an exact page title or person/thing name; returns the first sentences of that page",
    "lookup":    "give one keyword; returns the next sentence on the current page containing it",
    "calculate": "give arithmetic like 1966 - 1924; returns the result",
}

def squash(s):
    return " ".join(s.lower().split())


def run_react_demo():
    
    system_prompt = render('system/react.j2', tools=TOOL_DESCRIPTIONS)
    
    data = load_hot_qa()
    results = []
    for id in range(len(data)):
        
        row = data[id]
        
        question = row['question']
        context = row['context']
        gold = row['answer']
        
        tools = make_hotpot_tools(example=context)
        
        user_prompt = render('user/react.j2', question=question)
        resp = run_react(llm, system_prompt, user_prompt, parser=parse_react, tools=tools, max_steps=8, allowed_repeats={"lookup"})

        results.append(score_response(id, gold, resp, row['context']['title']))
    
    df = pd.DataFrame(results)
    
    df.to_csv(RESULTS_DIR / "react.csv", index=False, encoding='utf-8')
    
    sum = summary(df)
    
    (RESULTS_DIR / "react.json").write_text(json.dumps(sum, indent=2), encoding='utf-8')
    
def summary(df):
    
    # "methods": {str(k): int(v) for k, v in df["method"].fillna("none").value_counts().items()}
    status_counts = {str(k): int(v) for k, v in df['status'].fillna("none").value_counts().items()}
    contains_no_fallback = len(df.loc[(df['contains']) & ~(df['used_fallback'])]) / len(df)
    contains_with_fallback = len(df.loc[(df['contains']) & (df['used_fallback'])]) / len(df)
    
    tools = {str(k): int(v) for k, v in df["tools_used"].explode().value_counts().items()}
    
    summary = {'id': len(df), 'em': float(df['em'].mean()), 'f1': float(df['f1'].mean()), 'contains': float(len(df.loc[df['contains']]) / len(df)),
               'seconds_per_question': float(df['seconds'].mean()), 'avg_steps': float(df['n_steps'].mean()), 'status_counts': status_counts,
               'contains_no_fallback': float(contains_no_fallback), 'contains_with_fallback': contains_with_fallback, 'hit_rate': df['title_hit_rate'].mean(),
               'tools_used': tools}
    
    return summary
    
def search_hit(searched_title, titles):
    
    title_mapping = {titles[i].lower(): titles[i] for i in range(len(titles))}

    norm_title = searched_title.lower().strip()
    title = title_mapping.get(norm_title)
    
    if title is None:
        title = difflib.get_close_matches(norm_title, title_mapping.keys(), n=1, cutoff=0.5)
        
        return len(title) != 0
    
    return True
        
def score_response(id, gold, resp, titles):
    
    exact = None
    f1_score = None
    contains = False
    if resp.answer is not None:
        exact = exact_match(gold, resp.answer)
        f1_score = f1(gold, resp.answer)
        
        if squash(gold) in squash(resp.answer):
            contains = True
    
    tools_used = []
    titles_searches = []
    latency = 0
    for step in resp.steps:
        
        if step.tool is not None:
            tools_used.append(step.tool)

            if step.tool == "search":
                titles_searches.append(search_hit(step.arg, titles))

        latency += step.latency_s
    
    hits = float(len([t for t in titles_searches if t]) / len(titles_searches)) if len(titles_searches) > 0 else None

    result = {'id': id, 'gold': gold, 'answer': resp.answer, 'em': exact, 'f1': f1_score, 'contains': contains, 'status': resp.status, 'used_fallback': resp.used_fallback,
              'n_steps': len(resp.steps), 'tools_used': tools_used, 'title_hit_rate': hits, 'seconds': latency}

    return result
        

if __name__ == "__main__":
    
    #run_react_demo()
    
    df = pd.read_csv(RESULTS_DIR / 'react.csv', encoding='utf-8')
    
    print(df.loc[df['used_fallback']])