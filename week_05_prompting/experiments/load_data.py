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

def banking77_labels() -> list[str]:
    """All 77 intents from the full file — never from a sample."""
    return sorted(pd.read_csv(DATA_PATH / "banking77_train.csv")["label_name"].unique())
    