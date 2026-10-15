from chunking.base import Chunker, Chunk, make_chunk, trim_span, count_tokens
from chunking.fixed import FixedSizeChunker
import re

PAR_SEP = re.compile(r"\r?\n(?:[ \t]*\r?\n)+")
LINE_SEP = re.compile(r"\r?\n")

class RecursiveChunker(Chunker):
    
    def __init__(self, tokenizer, splitter, budget: int, min_tokens: int = 0, merge_across_paragraphs: bool = False):
        
        if budget <= 0:
            raise ValueError("ERROR: Budget must be greater than 0.")
        if min_tokens < 0 or min_tokens > budget:
            raise ValueError("ERROR: Invalid min_tokens value. Must be greater than 0 and less than the budget")
        
        self.tok = tokenizer
        self.splitter = splitter
        self.budget = budget
        self.min_tokens = min_tokens
        self.merge_across_paragraphs = merge_across_paragraphs
        
        self.fallback = FixedSizeChunker(tokenizer=tokenizer, size=budget, overlap=0)
    
    def _split(self, doc_text, start, end, level) -> list[tuple[int, int, int]]:
        
        start, end = trim_span(doc_text=doc_text, start=start, end=end)
        
        if start == end:
            return []
        
        tokens = count_tokens(tokenizer=self.tok, text=doc_text[start:end])
        
        if tokens <= self.budget:
            return [tuple([start, end, tokens])]
        
        if level == 1:
            
            lins = []
            
            curr_start = start
            for m in LINE_SEP.finditer(doc_text, start, end):
                
                start = m.start()
                
                lins.append([curr_start, start])
                
                curr_start = m.end()
            
            lins.append([curr_start, end])
            
            if len(lins) == 1:
                return self._split(doc_text, start, end, level = level + 1)
            else:
                results = []
                
                for l in lins:
                    
                    results.extend(self._split(doc_text=doc_text, start=l[0], end=l[1], level= level + 1))
                
                return results
        
        if level == 2:
            
            splits = self.splitter(doc_text[start:end])
            
            if len(splits) == 1:
                return self._split(doc_text=doc_text, start=start, end=end, level= level + 1)
            else:
                results = []
                for split in splits:
                
                    results.extend(self._split(doc_text=doc_text, start=start + split[0], end=start + split[1], level= level + 1))
                
                return results
        
        if level == 3:
            
            chunks = self.fallback.chunk(doc_text[start:end], doc_id="")
            results =[]
            
            for chunk in chunks:
                results.append(tuple([start + chunk.start, start + chunk.end, chunk.n_tokens]))
            
            return results
    
           
    def chunk(self, doc_text, doc_id) -> list[Chunk]:
        
        paras = []
        
        curr_start = 0
        for m in PAR_SEP.finditer(doc_text):
            
            paras.append([curr_start, m.start()])
            
            curr_start = m.end()
        
        paras.append([curr_start, len(doc_text)])
        
        blocks = []
        
        for i in range(len(paras)):
            
            res = self._split(doc_text, start=paras[i][0], end=paras[i][1], level=1)
            
            if len(res) == 0:
                continue
            
            for r in res:
                
                if len(r) == 0:
                    continue
                
                blocks.append(tuple([r[0], r[1], r[2], i]))
        
        if len(blocks) == 0:
            return []        
        
        
        start = 0
        end = 0
        curr_tokens = 0
        para_tag = 0
        chunks = []
        
        curr = 0
        while curr < len(blocks):
            
            block = blocks[curr]
            
            if not self.merge_across_paragraphs and para_tag != block[3]:
                
                if curr_tokens < self.min_tokens and curr_tokens + block[2] <= self.budget:
                    curr_tokens += block[2]
                    end = block[1]
                    para_tag = ""
                    curr += 1
                else:
                    if start != end:
                        chunk = make_chunk(doc_text=doc_text, doc_id=doc_id, chunk_index=len(chunks), start=start, end=end, tokenizer=self.tok)
                        chunks.append(chunk)
                
                    start = block[0]
                    end = block[1]
                    curr_tokens = block[2]
                    para_tag = block[3]
                    curr += 1
            
            else:
                
                if curr_tokens + block[2] <= self.budget:
                
                    curr_tokens += block[2]
                    end = block[1]
                
                    curr += 1
                    
                else:
                    
                    chunk = make_chunk(doc_text=doc_text, doc_id=doc_id, chunk_index=len(chunks), start=start, end=end, tokenizer=self.tok)
                    chunks.append(chunk)
                    
                    start = block[0]
                    end = block[1]
                    curr_tokens = block[2]
                    para_tag = block[3]
                    
                    curr += 1
                
        # There will always be a last chunk
        last_chunk = make_chunk(doc_text=doc_text, doc_id = doc_id, chunk_index=len(chunks), start = start, end= end, tokenizer=self.tok)
        
        chunks.append(last_chunk)
        
        return chunks