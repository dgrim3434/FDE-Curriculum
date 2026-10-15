from dataclasses import dataclass
from bisect import bisect_right
import re

@dataclass
class Heading:
    level: int
    title: str
    start: int

@dataclass
class Section:
    level: int
    title: str
    path: list[str]
    start: int
    end: int


MD_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*?)\s*$")
CHAPTER_RE = re.compile(r"^(CHAPTER|Chapter|BOOK|Book|PART|Part)\s+([IVXLCDM]+|\d+|[A-Za-z]+)\b.*$")
KNOWN = {"abstract", "introduction", "background", "related work", "methods", "materials and methods",
         "results", "discussion", "conclusion", "conclusions", "references", "acknowledgments",
         "acknowledgements"}

def detect_headings(doc_text) -> list[Heading]:
    
    curr_pos = 0
    results = []
    for raw in doc_text.splitlines(keepends=True):
        
        line = raw.rstrip("\r\n")
        
        # Searching for Markdown
        m = MD_RE.match(line)
        #breakpoint()
        if m is not None:
            level = len(m.group(1))
            title = m.group(2)
            results.append(Heading(level=level, title=title.strip(), start=curr_pos))
            curr_pos += len(raw)
            continue
        
        # Searching for numbered headers
        m = NUM_RE.match(line)
        
        if m is not None:
            
            if len(m.group(2).split()) > 10 or m.group(2).endswith("."):
                curr_pos += len(raw)
                continue
            
            level = m.group(1).count(".") + 1
            title = m.group(2)
            
            results.append(Heading(level=level, title=title.strip(), start=curr_pos))

            curr_pos += len(raw)
            continue
        
        m = CHAPTER_RE.match(line)
        
        if m is not None:
            
            if len(line.split()) > 8:
                curr_pos += len(raw)
                continue
            level_1 = set(['BOOK', 'PART'])

            if m.group(1).upper() in level_1:
                
                results.append(Heading(level=1, title=line, start=curr_pos))
            else:
                results.append(Heading(level=2, title=line, start=curr_pos))
            
            curr_pos += len(raw)
            continue
        
        norm = line.strip().rstrip(".:").casefold()
        if norm in KNOWN:

            results.append(Heading(level=1, title=norm, start=curr_pos))
            
        curr_pos += len(raw)
    
    return results

def build_sections(headings, doc_len) -> list[Section]:
    
    if len(headings) == 0:
        return []
    
    stack = []
    results = []
    for head in headings:
            
        while len(stack) !=0 and stack[-1].level >= head.level:
            
            curr = stack.pop()
            curr.end = head.start
        
        path = [h.title for h in stack] + [head.title]
        
        
        sed = Section(level=head.level, title=head.title, path=path, start=head.start, end=None)
        results.append(sed)
        stack.append(sed)

    for head in stack:
        
        head.end = doc_len
    
    return results

def section_path_at(section, pos) -> list[str]:
    
    starts = [s.start for s in section]
    
    i = bisect_right(starts, pos) - 1
    
    if i == -1:
        return []
    
    return section[i].path
