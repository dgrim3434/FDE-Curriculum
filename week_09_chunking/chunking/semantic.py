import numpy as np
from chunking.base import Chunker, Chunk, count_tokens, trim_span, make_chunk
from chunking.sentence import SentenceChunker

def gap_similarities(E, window=1):
    
    if window == 1:
        
        return np.diag(E[:-1] @ E[1:].T)
    else:
        
        results = np.zeros(len(E) - 1)
        
        
        for i in range(len(E) - 1):
            
            #breakpoint()
            l_lower = max(0, i - window + 1)
            
            left = np.mean(E[l_lower: i + 1], axis = 0)
            right = np.mean(E[i + 1: i + window + 1], axis= 0)
            
            left_norm = left / np.maximum(np.linalg.norm(left), 1e-12)
            right_norm = right / np.maximum(np.linalg.norm(right), 1e-12)
            
            results[i] = float(left_norm.dot(right_norm))
        
        return results

def percentile_threshold(distances, percentile) -> float:

    
    return float(np.percentile(distances, percentile))


def cut_after(distances, threshold) -> list[int]:

    return np.where(distances > threshold)[0].tolist()


class SemanticChunker(Chunker):
    
    def __init__(self, tokenizer, splitter, embed_fn, budget: int, threshold: float | None = None, window: int = 1, min_tokens: int = 0):
        
        if budget <= 0:
            raise ValueError("ERROR: The budget must be atleast 1")
        if window < 1:
            raise ValueError("ERROR: Window Must be atleast 1")
        if min_tokens < 0 or min_tokens > budget:
            raise ValueError("ERROR: invalid min_tokens. Must be atleast zero and less than the budget")
        
        self.tok = tokenizer
        self.splitter = splitter
        self.embed_fn = embed_fn
        self.budget = budget
        self.threshold = threshold
        self.window = window
        self.min_tokens = min_tokens
        self.sentences = SentenceChunker(tokenizer=tokenizer, splitter=splitter, budget=budget, overlap_sentences=0)
    
    def prepare(self, doc_text) -> tuple[list[tuple[int, int, int]], np.array]:
        
        splits = self.splitter(doc_text)
        
        sentences = []
        sentence_text = []
        for split in splits:
            
            start, end = trim_span(doc_text, split[0], split[1])
            
            if start == end:
                continue
            
            tokens = count_tokens(self.tok, text=doc_text[start: end])
            
            sentences.append(tuple([start, end, tokens]))
            sentence_text.append(doc_text[start:end])
        
        if len(sentences) < 2:
            
            return tuple([sentences, np.array([])])
        
        E = self.embed_fn(sentence_text)
        
        E_norm = E / np.maximum(np.linalg.norm(E, axis = 1, keepdims=True), 0)
        
        similarities = gap_similarities(E_norm, window=self.window)
        
        distances = 1 - similarities
        
        return tuple([sentences, distances])

    
    def chunk_prepared(self, doc_text, doc_id, sentences, distances) -> list[Chunk]:
        
        if self.threshold is None:
            raise ValueError("ERROR: threshold cannot be none")

        
        cuts = cut_after(distances, self.threshold)
        
        segments = []
        
        curr_start = 0
        
        for cut in cuts:
            
            start = sentences[curr_start]
            end = sentences[cut]
            tokens = count_tokens(tokenizer=self.tok, text=doc_text[start[0]: end[1]])
            segments.append([curr_start, cut, tokens])
            curr_start = cut + 1 
        
        tokens = count_tokens(tokenizer=self.tok, text=doc_text[sentences[curr_start][0]:])
        segments.append([curr_start, len(sentences) - 1, tokens])
        
        merged = []
        
        seg = 0
        while seg < len(segments):
            
            segment = segments[seg]
            if segment[2] < self.min_tokens:
                
                if len(merged) == 0 or merged[-1][2] + segment[2] > self.budget:
                    
                    if seg + 1 == len(segments) or segments[seg + 1][2] + segment[2] > self.budget:
                        merged.append(segment)
                        seg += 1
                    else:
                        start = segment[0]
                        end = segments[seg + 1][1]
                        tokens = segments[seg + 1][2] + segment[2]
                        merged.append([start, end, tokens])
                        seg += 2
                        
                elif seg + 1 == len(segments) or segments[seg + 1][2] + segment[2] > self.budget:
                    
                    end = segment[1]
                    updated_toks = merged[-1][2] + segment[2]
                    merged[-1] = [merged[-1][0], end, updated_toks]
                    seg += 1
                else:
                    
                    left = distances[merged[-1][1]]
                    right = distances[segment[1]]
                    
                    if left < right:
                        merged[-1] = [merged[-1][0], segment[1], merged[-1][2] + segment[2]]
                        seg += 1
                    else:
                        
                        merged.append([segment[0], segments[seg + 1][1], segment[2] + segments[seg + 1][2]])
                        seg += 2    
            else:
                merged.append(segment)
                seg += 1

        chunks = []

        for merge in merged:
            #breakpoint()
            if merge[2] > self.budget:
                
                cs = self.sentences.chunk(doc_text[sentences[merge[0]][0]: sentences[merge[1]][1]], doc_id)
                start = sentences[merge[0]][0]
                for c in cs:
                    
                    
                    chunk = make_chunk(doc_text, doc_id, chunk_index=len(chunks), start= start + c.start, end= start + c.end, tokenizer=self.tok)

                    chunks.append(chunk)
            else:
                #breakpoint()
                chunk = make_chunk(doc_text, doc_id, chunk_index=len(chunks), start=sentences[merge[0]][0], end = sentences[merge[1]][1], tokenizer=self.tok)
                    
                chunks.append(chunk)
        
        return chunks

    def chunk(self, doc_text, doc_id):
        
        sentences, distance = self.prepare(doc_text)
        
        if len(sentences) == 0:
            return []
        chunks = self.chunk_prepared(doc_text, doc_id, sentences, distance)

        return chunks