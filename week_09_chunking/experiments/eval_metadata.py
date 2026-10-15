from pathlib import Path
from metadata.fields import extract_fields
import json
import pandas as pd


DATA = Path(__file__).parent.parent / "data/gutenberg"
RESULTS = Path(__file__).parent.parent / "results"
label_path = DATA / "labels.json"


with open(label_path, 'r', encoding='utf-8') as f:
    
    data = json.load(f)


def norm_title(s):
    
    if s is None:
        return None
        
    return " ".join(s.casefold().split())

def catalog_surnames(author_str):
    
    if author_str is None:
        return None
    splits = author_str.split(";")
    results = set()
    
    for s in splits:
        
        if "[" in s:
            continue
        
        surname = s.split(",")[0]
    
        results.add(surname.strip().casefold())
    
    return results  

def extracted_surnames(names):
    
    results = set()
    if names is None:
        return None
    for name in names:
        
        surname = name.split()[-1]
        
        results.add(surname.strip().casefold())       
    return results
    
def run_experiment():
    
    results = []
    answers = []
    for id, values in data.items():
        result = {}
        result['id'] = id
        text_path = DATA / f"pg{id}.txt"
        text = text_path.read_text(encoding='utf-8')

        extracted = extract_fields(text)
        
        if extracted['title'] is None:
            result['title_extracted'] = False
            result['title_correct'] = False
        else:
            result['title_extracted'] = True
            result['title_correct'] = norm_title(values['title']) == norm_title(extracted['title'])
        
        if extracted['authors'] is None:
            result['authors_extracted'] = False
            result['authors_correct'] = False
        else:
            result['authors_extracted'] = True
            author_correct = catalog_surnames(values['authors']) == extracted_surnames(extracted['authors'])
            result['authors_correct'] = author_correct
            
        if extracted['date'] is None:
            result['date_extracted'] = False
            result['date_correct'] = False
        else:
            result['date_extracted'] = True
            result['date_correct'] = extracted['date'] == values['issued']
        
        answer = {'correct_title': values['title'], 'predicted_title': extracted['title'], 'correct_author': values['authors'], 'predicted_author': extracted['authors'],
                  'correct_date': values['issued'], 'predicted_date': extracted['date']}
        answers.append(answer)
        results.append(result)
    
    out_dir = RESULTS / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir /"gutenberg_fields_raw.json").write_text(json.dumps(results), encoding='utf-8')
    (out_dir / "gutenberg_answers.json").write_text(json.dumps(answers), encoding='utf-8')
    summary = generate_summary(results)
    
    (out_dir / "gutenberg_fields_summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
        
def generate_summary(results):
    
    df = pd.DataFrame(results)
    
    summary = {}
    
    for field in ['title', 'authors', 'date']:
        
        correct = len(df.loc[df[f'{field}_correct']])
        coverage = float(len(df.loc[df[f'{field}_extracted']]) / len(df))
        accuracy = float( correct / len(df))
        accuracy_covered = float(correct / len(df.loc[df[f'{field}_extracted']])) if len(df.loc[df[f'{field}_extracted']]) > 0 else None
        
        summary[f'{field}_coverage'] = coverage
        summary[f'{field}_accuracy'] = accuracy
        summary[f'{field}_accuracy_covered'] = accuracy_covered
    
    return summary
        


if __name__ == "__main__":
    run_experiment()
    
    