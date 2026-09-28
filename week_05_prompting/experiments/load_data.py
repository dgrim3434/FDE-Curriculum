import pandas as pd
from pathlib import Path
import numpy as np
DATA_PATH = Path(__file__).resolve().parent.parent / "data"


def load_banking77(training_size=1000, testing_size=200, seed = 42):
    
    train_df = pd.read_csv(DATA_PATH / "banking77_train.csv", encoding='utf-8')
    test_df = pd.read_csv(DATA_PATH / "banking77_test.csv", encoding='utf-8')
    
    train = train_df.sample(min(training_size, len(train_df)),random_state=seed)
    test = test_df.sample(min(testing_size, len(test_df)), random_state=seed)
    
    return train, test

def load_gsm8k(training_size = 1000, testing_size = 200, seed = 42):
    
    train_df = pd.read_csv(DATA_PATH / "gsm8k_train.csv", encoding="utf-8")
    test_df = pd.read_csv(DATA_PATH / "gsm8k_test.csv", encoding='utf-8')
    
    train = train_df.sample(min(training_size, len(train_df)), random_state=seed)
    test = test_df.sample(min(testing_size, len(test_df)), random_state=seed)
    
    return train, test

def random_sampling(data, size, seed = 42):
    
    rng = np.random.default_rng(seed=seed)
    
    
    if size >= len(data):
        return data
    
    if size <= 0:
        raise ValueError("ERROR: Sampling size must be atleast 1")
    
    
    idx = rng.integers(0, len(data), size=size)
    
    return data.iloc[idx].reset_index(drop='index')
    