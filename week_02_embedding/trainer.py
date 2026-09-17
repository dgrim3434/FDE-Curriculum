import numpy as np
from vocab_preprocessing import Vocabulary
from sampler import Sampler
from skip_gram import Skip_Gram
import math
from analysis import Embedding_Analyzer
class Trainer:
    
    def __init__(self, corpus_path, dimensions = 100, max_vocab = 15000, min_count = 5, w=5, k=5, epochs = 5, batch_size = 128, learning_rate=0.025, min_learning_rate=0.0001,t = 1e-4, logging_rate= 1000, seed=0):
        
        self.k = k
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate 
        self.min_learning_rate = min_learning_rate
        self.logging_rate = logging_rate
        self.seed = seed
        self.dimensions = dimensions
        self.vocabulary = Vocabulary(corpus_path)
        self.vocabulary.build_vocabulary(max_vocab, min_count)
        self.w = w
        self.t = t
        self.text = self.vocabulary.convert_to_ids()
        
        self.sampler = Sampler(self.vocabulary.word_frequencies)
        self.sampler.negative_sampling()
        
        self.model = Skip_Gram(np.minimum(len(self.vocabulary.idx_to_word), max_vocab), self.dimensions)

    def train(self, save_path):
        
        rng = np.random.default_rng(self.seed)
        # Loss Function Check
        
        step = 0
        lr0 = self.learning_rate
        #epoch_text = self.vocabulary.subsample(self.text, t=self.t)

        for ep in range(self.epochs):
            total_loss = 0
            pos_loss = 0
            neg_loss = 0
            epoch_text = self.vocabulary.subsample(self.text, t=self.t)
            centers, context = self.sampler.compute_pairs(epoch_text, max_window=self.w)
            n = len(centers)
            steps_per_ep = n // self.batch_size
            if ep == 0:
                total_steps = steps_per_ep * self.epochs
            perm = rng.permutation(n)
            
            for st in range(steps_per_ep):
                upper = np.minimum(n, (st + 1) * self.batch_size)
                po = perm[st * self.batch_size: upper]
                
                c = centers[po]
                ctx = context[po]
                
                negs = self.sampler.negative_draw(len(ctx), self.k)
                
                targets = np.concatenate([ctx[:,None], negs], axis = 1)
                
                lr = lr0 - (lr0 - self.min_learning_rate) * (step / total_steps)

                scores, loss = self.model.step(c, targets, lr)
                if step == 0:
                    assert abs(loss - (self.k+1)*np.log(2)) < 1e-3, "LOSS FUNCTION IS WRONG"
                    assert np.abs(scores).max() == 0.0, "Scores should be zero for first step"
                    
                total_loss += loss
                pos_loss += scores[:, 0].mean()
                neg_loss += scores[:, 1:].mean()
                
                #print(f"Step: {step} finished out of: {total_steps}")
                step += 1
                if step % self.logging_rate == 0:
                    print(f"ep{ep} step{step}/{total_steps} lr={lr:.5f} "
                        f"loss={total_loss/self.logging_rate:.4f} "
                        f"pos={pos_loss/self.logging_rate:+.3f} neg={neg_loss/self.logging_rate:+.3f} "
                        f"|W_in|={np.linalg.norm(self.model.w_in,axis=1).mean():.3f}")
                    total_loss = pos_loss = neg_loss = 0.0
            
            print(f"Epoch {ep} Completed. Most Epoch Report:\n")
            
            analy = Embedding_Analyzer(self.model.w_in, self.vocabulary.word_to_idx, self.vocabulary.idx_to_word)
            for w in ["king", "man", "water", "one", "city"]:
                try:
                    print(f"Word: {w}")
                    print(analy.most_similar(w, k=3))
                except Exception:
                    print(f"{w} not in corpus")
        self.model.save(save_path)