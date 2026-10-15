from chunking.base import Chunk, make_chunk, Chunker
import math


class FixedSizeChunker(Chunker):
    
    def __init__(self, tokenizer, size: int, overlap: int = 0):
        
        self.tok = tokenizer
        self.size = size
        self.overlap = overlap
        
        if overlap >= size:
            raise ValueError(f"ERROR: size: {size}, must be larger than overlap: {overlap}")

        if overlap < 0:
            raise ValueError(f"ERROR: overlap must be atleast 0")
    def chunk(self, doc_text, doc_id) -> list[Chunk]:
        
        chunks = []
        
        enc = self.tok(doc_text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        
        offsets = enc["offset_mapping"]
        
        if len(offsets) == 0:
            return []
        
        stride = self.size - self.overlap
        
        steps = math.ceil(len(offsets) / stride)
        for i in range(steps):

            
            start_pos = i * stride
            end_pos = min(len(offsets), start_pos + self.size)
            
            start = offsets[start_pos][0]
            end = offsets[end_pos - 1][1]
            
            chunk = make_chunk(doc_text, doc_id, i, start, end, self.tok)
            
            if chunk is not None:
                chunks.append(chunk)

            if end_pos == len(offsets):
                break
        return chunks