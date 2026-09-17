"""
Vocabulary Preprocessing:

Responsible for getting the vocabulary in the correct format for testing and building both vocab dictionaries:
    - Ingesting the Vocabulary from the specified path
    - Splitting into words and keeping the top V (V represents the vocabulary size)
    - The Subsampling Function
"""
from collections import Counter
import numpy as np
import re

class Vocabulary:
    
    def __init__(self, corpus):
        
        self.corpus = corpus
        self.vocab_size = None
        self.idx_to_word = None
        self.word_to_idx = None
        self.text = None
        self.train_corpus = None
        self.word_frequencies = None
        with open(self.corpus, 'r', encoding='utf-8') as f:
            self.text = f.read()
            text_len = np.minimum(30_000_000, len(self.text))
            self.text = self.text[:text_len]
            
           # self.text = re.findall(r"[a-z]+",self.text.lower())
            
        
        
    def build_vocabulary(self, max_vocab, min_count):
        
        # Creating an array of all of the words within the corpus
        words = re.findall(r"[a-z]+",self.text.lower())
        #words = self.text.split()
        word_counts = Counter(words)
        
        # Remove all words with less than 5 occurances
        kept_words = [(w,c) for w,c in word_counts.items() if c >= min_count][:max_vocab]

        self.idx_to_word = np.array([w for w, _ in kept_words])
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
        word_count = [word_counts[w] for w in self.idx_to_word]
        self.word_frequencies = np.array(word_count) / sum(word_count)
        self.vocab_size = len(self.idx_to_word)
        
    def convert_to_ids(self):
        
        if self.text == None:
            raise ValueError("ERROR: The test must be ingested before calling function")
        
        if self.word_to_idx == None:
            raise ValueError("ERROR: build_vocabulary function must be called before converting text to ids")
        
        words = re.findall(r"[a-z]+",self.text.lower())
        
        word_ids = np.array([self.word_to_idx[w] for w in words if w in self.word_to_idx])
        
        return word_ids
    
    def subsample(self, word_ids, t=None, rng=None):
        
        if t == None:
            return word_ids
        
        # if self.word_frequencies == None:
        #     raise ValueError("ERROR: The Vocabulary Must be built before subsampling")
        
        frequencies = self.word_frequencies
        
        keep_prob = np.minimum(np.sqrt(t / frequencies) + t / frequencies, 1)
        
        rng = rng or np.random.default_rng()
        
        self.train_corpus = word_ids[rng.random(len(word_ids)) < keep_prob[word_ids]]
        
        
        return self.train_corpus