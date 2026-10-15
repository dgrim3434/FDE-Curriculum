from dataclasses import dataclass, field

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    start: int
    end: int
    text: str
    n_tokens: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        
        return f"{self.doc_id}::{self.chunk_index}"


def count_tokens(tokenizer, text) -> int:
    
    if len(text) == 0:
        return 0
    
    return len(tokenizer(text, add_special_tokens = False)['input_ids'])
    
def trim_span(doc_text, start, end) -> tuple[int, int]:
    
    while start < end and doc_text[start].isspace():
        start += 1
    
    end = end - 1
    while end > start and doc_text[end].isspace():
        end -= 1
    
    if start > end:
        
        return (start, start)
    
    return (start, end + 1)

def make_chunk(doc_text, doc_id, chunk_index, start, end, tokenizer) -> Chunk | None:
    
    start, end = trim_span(doc_text, start, end)
    
    if start == end:
        return None
    
    text = doc_text[start: end]
    
    n_tokens = count_tokens(tokenizer=tokenizer,text=text)
    
    return Chunk(doc_id=doc_id, chunk_index=chunk_index, start=start, end=end, text=text, n_tokens=n_tokens)


class Chunker:
    
    def chunk(self, doc_text: str, doc_id: str) -> list[Chunk]:
        
        raise NotImplementedError