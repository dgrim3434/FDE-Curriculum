import re
from datetime import datetime


START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ]*?):\s*(.*)$")
BRACKET_RE = re.compile(r"\s*\[.*?\]\s*$")
HEADER_LINE_RE = re.compile(r"The Project Gutenberg e-?book of (.+?)(?:, by .*)?$", re.IGNORECASE)

def front_matter(text) -> str:
    
    m = START_RE.search(text)
    
    if m is None:
        return "\n".join(text.splitlines()[:100])
    
    return text[:m.start()]

def parse_fields(front) -> dict[str, list[str]]:
    
    last_label = None
    results = {}
    
    for line in front.splitlines():
        
        if len(line) == 0 or line.isspace():
            continue
        
        m = LABEL_RE.match(line)
        
        if m is not None:
            label = m.group(1).strip().lower()
            value = m.group(2).strip()
            
            if label not in results:
                results[label] = [value]
                last_label = label
            else:
                last_label = None
            
            
            continue
        
        if line[:1].isspace() and last_label is not None:
            
            results[last_label].append(line.strip())
        else:
            
            last_label = None
    
    return results

FORMATS = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y"] 

def parse_date(raw) -> str | None:
    
    date = BRACKET_RE.sub("", raw).strip()
    
    for fmt in FORMATS:
        
        try:
            
            return datetime.strptime(date, fmt).date().isoformat()
        
        except ValueError:
            continue
    
    
    try:
        d = datetime.strptime(date, "%B %Y")
        return d.strftime("%Y-%m")
    except ValueError:
        return None

def extract_fields(text) -> dict:
    
    front = front_matter(text)
    fields = parse_fields(front)
    
    results = {}
    results['provenance'] = {}
    if 'title' in fields:
        
        results['title'] = " ".join(fields.get('title'))
        results['provenance']['title'] = 'label:title'
        
    else:
        #breakpoint()
        for line in text.splitlines():
            
            if not line.isspace() and len(line) != 0:
                
                m = HEADER_LINE_RE.match(line)
                if m is not None:
                    results['title'] = m.group(1).strip()
                    results['provenance']['title'] = "header_line"
                
                else:
                    results['title'] = None
                    results['provenance']['title'] = 'not_found'
                
                break
    
    if 'author' in fields:
        
        auths = []
        
        for auth in fields['author']:
            
            splits = auth.split(' and ')
            
            for s in splits:
                
                if len(s.strip()) != 0:
                    auths.append(s.strip())
        
        results['authors'] = auths
        results['provenance']['authors'] = 'label:author'
    else:

        results['authors'] = None
        results['provenance']['authors'] = 'not_found'
    
    for label in ['release date', 'posting date']:
        
        if label in fields:
            
            raw_date = fields[label][0]
            
            date = parse_date(raw_date)
            
            results['date'] = date
            results['provenance']['date'] = f'label:{label}'
            results['date_raw'] = raw_date
            break
    
    if 'date' not in results:
           
        results['date'] = None
        results['date_raw'] = None
        results['provenance']['date'] = 'not_found'
    return results