from trainer import Trainer
from analysis import Embedding_Analyzer
import numpy as np

np.seterr(divide='raise', over='raise', invalid='raise', under='ignore')
if __name__ == "__main__":
    
    train = Trainer("corpus/prose_text.txt")
    train.train("models/prose_model")
    
    
    
    
    
    