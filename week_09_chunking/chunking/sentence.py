from chunking.base import Chunker, Chunk, make_chunk, count_tokens, trim_span
from chunking.fixed import FixedSizeChunker

class SentenceChunker(Chunker):
    
    def __init__(self, tokenizer, splitter, budget: int, overlap_sentences: int = 0):
        
        self.tok = tokenizer
        self.splitter = splitter
        self.budget = budget
        self.overlap = overlap_sentences
        
        if budget <= 0:
            raise ValueError("ERROR: Budget must be greater than 0")
        
        if overlap_sentences < 0:
            raise ValueError("ERROR: Sentence overlap must be atleast 0")
        
        self.fallback = FixedSizeChunker(tokenizer=tokenizer, size=budget, overlap=0)
    
    def chunk(self, doc_text, doc_id) -> list[Chunk]:
        
        splits = self.splitter(doc_text)
        
        sentences = []
        
        for split in splits:
            
            span = trim_span(doc_text, split[0], split[1])
            
            if span[0] == span[1]:
                continue
            
            tokens = count_tokens(self.tok, doc_text[span[0]:span[1]])
            
            sentences.append((span[0], span[1], tokens))
        
        if len(sentences) == 0:
            return []
        
        current = []
        curr_budget = 0
        curr_chunk = 0
        curr_overlap = 0
        chunks = []
        i = 0
        while i < len(sentences):
            
            if curr_budget + sentences[i][2]  <= self.budget:
                #breakpoint()
                curr_budget += sentences[i][2]
                
                current.append(sentences[i])
                i += 1
                
            else:
                
                if sentences[i][2] > self.budget:
                    
                    if len(current) > curr_overlap:
                        start = current[0][0]
                        end = current[-1][1]
                        chunk = make_chunk(doc_text, doc_id, chunk_index=curr_chunk, start=start, end = end, tokenizer=self.tok)
                        chunks.append(chunk)
                        curr_chunk += 1
                    
                    
                    cs = self.fallback.chunk(doc_text[sentences[i][0]: sentences[i][1]], doc_id)
                    start = sentences[i][0]
                    for c in cs:
                        
                        chunk = make_chunk(doc_text=doc_text, doc_id=doc_id, chunk_index=curr_chunk, start=start + c.start, end=start + c.end, tokenizer=self.tok,)
                        chunks.append(chunk)
                        curr_chunk += 1
                        
                    current = []
                    curr_overlap = 0
                    curr_budget = 0  
                    i += 1
                
                else:
                    
                    if len(current) > curr_overlap:
                        
                        start = current[0][0]
                        end = current[-1][1]
                        chunk = make_chunk(doc_text, doc_id, chunk_index=curr_chunk, start=start, end=end, tokenizer=self.tok)    
                        chunks.append(chunk)
                        curr_chunk += 1
                        
                        overlap = min(len(current), self.overlap)
                        
                        if overlap > 0:
                            overs = current[-overlap:]
                        
                            if overs[0][1] == current[0][1]:
                                overs.pop(0)
                                overlap -= 1
                        
                            current = overs
                            curr_budget = 0
                            curr_overlap = overlap
                            for j in overs:
                            
                                curr_budget += j[2]
                        else:
                            current = []
                            curr_budget = 0
                            curr_overlap = 0
                    else:
                        
                        while curr_budget + sentences[i][2] > self.budget:
                            
                            drop = current.pop(0)
                            curr_overlap -= 1
                            curr_budget -= drop[2]
                        
                        
                        current.append(sentences[i])
                        curr_budget += sentences[i][2]
                        i += 1
                    

        if len(current) > curr_overlap:
            start = current[0][0]
            end = current[-1][1]
            
            chunk = make_chunk(doc_text, doc_id, curr_chunk, start, end, self.tok)
            chunks.append(chunk)
            
        return chunks        
                        