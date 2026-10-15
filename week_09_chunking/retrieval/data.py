import json

def load_corpus(path) -> dict:
    
    results = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        
        for line in f:
            
            if not line.strip():
                continue
            
            row = json.loads(line)
            
            doc_id = str(row['_id'])
            
            title = row['title']
            
            text = row['text']
            
            results[doc_id] = {'title': title, 'text': text}
    return results

def load_queries(path):
    
    results = {}
    
    with open(path, encoding='utf-8') as f:
        
        for line in f:
            
            if not line.strip():
                continue
            row = json.loads(line)
            
            doc_id = str(row['_id'])
            text = row['text']
            
            results[doc_id] = text
    
    return results

def load_qrels(path):
    
    results = {}
    
    with open(path, encoding='utf-8') as f:
        next(f) # Skipping the header
        
        for line in f:
            
            qid, cid, score = line.rstrip("\n").split("\t")
            
            if qid not in results:
                results[str(qid)] = {}
                
            results[str(qid)][str(cid)] = int(score)
    
    return results

def load_beir(root, split="test") -> tuple[dict, dict, dict]:
    
    corp_path = root / "corpus.jsonl"
    query_path = root / "queries.jsonl"
    qrel_path = root / f"qrels/{split}.tsv"
    
    corp = load_corpus(corp_path)
    query = load_queries(query_path)
    qrel = load_qrels(qrel_path)
    
    return tuple([corp, query, qrel])