from tokenizer import Tokenizer

def train_tokenizer(text_corpus_path, vocab = 512):
    
    tokenizer = Tokenizer()
    
    with open(text_corpus_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tokenizer.train(vocab, content)
    
    # Log all merges
    file = f"merge_logs/merges_vocab_{vocab}"
    with open(file, 'w', encoding='utf-8') as f:
        
        for (p0, p1) in tokenizer.merged:
            merged_bytes = tokenizer.vocab[p0] + tokenizer.vocab[p1]
            
            f.write(f"IDs: {p0} {p1}, Vocab: {merged_bytes}, Decoding: '{merged_bytes.decode('utf-8', errors='replace')}'\n")
    
    
if __name__ == "__main__":
    path = "corpus/corpus_code.txt"
    
    vocab_size = [512, 1024, 4096]
    
    for vocab in vocab_size:
        print(f"Starting Training Vocab Size: {vocab}")
        train_tokenizer(path, vocab)
        print(f"Finished Training")