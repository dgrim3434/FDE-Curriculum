from vocab_preprocessing import Vocabulary
from skip_gram import Skip_Gram
from analysis import Embedding_Analyzer


def most_similar(eval, words):
    
    results = {}
    
    for w in words:
        try:
            results[w] = {}
        
            similar = eval.most_similar(w, k = 3)

            
            results[w] = similar
        except Exception:
            pass
    
    return results

def analogies(eval, words):
    result_1 = {}
    result_2 = {}
    for w in words:
        a, b, c = w
        
        try:
            anal = eval.analogies(a,b,c, k=3)
       
            result_1[w] = anal

            anal_2 = eval.analogies(a,b,c, strategy='mult', k=3)
            result_2[w] = anal_2
        except Exception:
            pass
    
    return result_1, result_2

if __name__ == "__main__":
    
    vocab_prose = Vocabulary("corpus/prose_text.txt")
    vocab_text = Vocabulary("corpus/text8")
    vocab_text.build_vocabulary(15_000, min_count=5)
    vocab_prose.build_vocabulary(4_206, min_count=5)
    
    model_prose = Skip_Gram(4206, 100)
    model_prose.load("models/prose_model")
    model_text = Skip_Gram(15_000, 100)
    model_text.load("models/text8_model")
    
    prose_anal = Embedding_Analyzer(model_prose.w_in, vocab_prose.word_to_idx,vocab_prose.idx_to_word)
    text8_anal = Embedding_Analyzer(model_text.w_in, vocab_text.word_to_idx, vocab_text.idx_to_word)
    
    prose_anal.visualizations("prose")
    text8_anal.visualizations("text8")
    PROBES = ["king","man","water","one","city","france","run","good","two","said"]
    
    prose_similar = most_similar(prose_anal, PROBES)
    text8_similar = most_similar(text8_anal, PROBES)
    
    ANALS = [('one', 'two', 'three'), ('run', 'running', 'walk'), ('dog', 'dogs', 'cat'), ('man', 'king', 'woman')]
    
    prose_anal_add, prose_anal_mult = analogies(prose_anal, ANALS)
    text8_anal_add, text8_anal_mult = analogies(text8_anal, ANALS)
    
    prose_baseline_sim = prose_anal.generate_random_pairs()
    text8_baseline_sim = text8_anal.generate_random_pairs()
    
    with open("artifacts/model_evals.txt", 'w', encoding='utf-8') as f:
        
        f.write(f"Prose Baseline Similarity: {prose_baseline_sim}\n")
        f.write(f"Text 8 Baseline Similarity: {text8_baseline_sim}\n")
        
        f.write(f"MODEL WORD SIMILARITY COMPARISON:\n\n")
        f.write("Prose Similarities:\n")
        for w, sims in prose_similar.items():
            f.write(f"word: {w}:\n")
            for sim in sims:
                f.write(f"      similar word: {sim[0]}, score: {sim[1]}\n")
        
        f.write("Text8 Similarities:\n")
        for w, sims in text8_similar.items():
            f.write(f"word: {w}:\n")
            for sim in sims:
                f.write(f"      similar word: {sim[0]}, score: {sim[1]}\n")
                
        
        f.write(f"MODEL WORD ANALOGIES COMPARISON\n\n")
        f.write(f"Prose Similarities (add)\n")
        for (a,b,c), sims in prose_anal_add.items():
            f.write(f"{b} is to {a} as {c} is to: ?\n")
            for sim in sims:
                f.write(f"      similar word: {sim[0]}, score: {sim[1]}\n")
        
        f.write(f"Text8 Similarities (add)\n")
        for (a,b,c), sims in text8_anal_add.items():
            f.write(f"{b} is to {a} as {c} is to: ?\n")
            for sim in sims:
                f.write(f"      similar word: {sim[0]}, score: {sim[1]}\n")
        
        f.write(f"Prose Similarities (mult)\n")
        for (a,b,c), sims in prose_anal_mult.items():
            f.write(f"{b} is to {a} as {c} is to: ?\n")
            for sim in sims:
                f.write(f"      similar word: {sim[0]}, score: {sim[1]}\n")

        f.write(f"Text8 Similarities (mult)\n")
        for (a,b,c), sims in text8_anal_mult.items():
            f.write(f"{b} is to {a} as {c} is to: ?\n")
            for sim in sims:
                f.write(f"      similar word: {sim[0]}, score: {sim[1]}\n")

