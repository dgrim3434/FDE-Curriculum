import json

def load_corpus(path):
    
    results = []
    with open(path, encoding='utf-8') as f:
        
        for line in f:
            
            row = json.loads(line)
            
            id = str(row['_id'])
            text = row['title'] + " " + row['text'].strip()
            
            results.append((id, text))
    
    return results

def load_queries(path):
    
    results = {}
    
    with open(path, encoding='utf-8') as f:
        
        for line in f:
            
            row = json.loads(line)
            
            id = str(row['_id'])
            text = row['text']
            
            results[id] = text
    
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
