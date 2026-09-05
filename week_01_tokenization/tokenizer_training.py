from tokenizer import Tokenizer

VOCAB_SIZES = [1024, 4096]

CORPUS = {
    'code': {'path': 'corpus/code_train.txt'},
    'prose': {'path': 'corpus/prose_train.txt'}
}

def train_tokenizer(corpus, vocab_size):
    print(f"Training Tokenizer on: {corpus} with vocab size: {vocab_size}")
    with open(CORPUS[corpus]['path'], 'r', encoding='utf-8') as f:
        text = f.read()
    
    token = Tokenizer()
    
    token.train(vocab_size, text)
    
    token.save(f"models/{corpus}_{vocab_size}")
    print(f"Model Saved to: models/{corpus}_{vocab_size}.model")


if __name__ == "__main__":
    
    for name in CORPUS:
        for vocab in VOCAB_SIZES:
            
            train_tokenizer(name, vocab)