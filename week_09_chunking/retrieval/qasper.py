

def render_paper(paper) -> tuple[str, list[tuple[int, str]]]:
    
    blocks = []
    rendered = []
    abstract = paper['abstract']
    
    if abstract is not None and len(abstract.strip()) > 0:
        blocks.append("# Abstract")
        blocks.append(abstract)
        rendered.append(tuple([1, "Abstract"]))
    
    for section in paper['full_text']:
        
        section_name = section['section_name']
        paragraph = section['paragraphs']
          
        if section_name is not None and len(section_name.strip()) > 0:
            parts = [p.strip().rstrip(".") for p in section_name.split(":::")]
            title = " > ".join(parts)
            blocks.append(f"# {title}")
            rendered.append(tuple([1, title]))
        
        for p in paragraph:
            if p is not None and len(p.strip()) > 0:
                blocks.append(p)
            
        
        
    
    return "\n\n".join(blocks), rendered

def qasper_doc_fields(paper) -> dict:
    
    title = paper['title']
    
    return {'title': title, 'authors': None, 'date': None, 'date_raw': None, 'provenance': {'title': 'source_field', 'authors': 'not_found', 'date': 'not_found'}}

def locate_evidence(doc_text, evidence) -> tuple[list[tuple[int, int]], dict]:
    
    spans = []
    counts = {"found": 0, "float_selected": 0, "not_found": 0}
    
    for ev in evidence:
        
        if ev.startswith("FLOAT SELECTED"):
            
            counts['float_selected'] += 1
            continue
        
        i = doc_text.find(ev)
        
        if i != -1:
            
            spans.append(tuple([i, i + len(ev)]))
            counts['found'] += 1
            continue
        
        s = ev.strip()
        
        i = doc_text.find(s)
        
        if i != -1:
            spans.append(tuple([i, i + len(s)]))
            counts['found'] += 1
            continue
        
        counts['not_found'] += 1
    
    return spans, counts

def qasper_questions(paper, doc_text) -> tuple[list[dict], dict]:
    
    counts = {'n_questions': 0, "found": 0, "float_selected": 0, "not_found": 0, 'n_unanswerable': 0, 'n_no_evidence_located': 0}
    questions = []
    for qa in paper['qas']:
        
        gold = []
        counts['n_questions'] +=1
        
        for a in qa['answers']:
            
            ans = a['answer']
            if ans['unanswerable']:
                continue
            
            spans, cs = locate_evidence(doc_text, ans['evidence'])
            
            counts['found'] += cs['found']
            counts['float_selected'] += cs['float_selected']
            counts['not_found'] += cs['not_found']
            
            if len(spans) > 0:
                gold.append(spans)
        
        if len(gold) != 0:
            questions.append({'question_id': qa['question_id'], 'question': qa['question'], 'gold': gold})
            
        else:
            
            if all(a['answer']['unanswerable'] for a in qa['answers']):
                counts['n_unanswerable'] += 1
            else:
                counts['n_no_evidence_located'] += 1
    
    return questions, counts