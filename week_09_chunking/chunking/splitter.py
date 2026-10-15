
def make_spacy_splitter(nlp):
    
    def run(doc_text):
        
        return [(s.start_char, s.end_char) for s in nlp(doc_text).sents]
    
    return run
    
