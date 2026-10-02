from pathlib import Path
import pandas as pd
import random
import json

DATA_DIR = Path(__file__).parent.parent / "data"

def load_gsm8k(train=100, dev=100, test=200, seed = 42):
    
    train_df = pd.read_csv(DATA_DIR / "gsm8k_train.csv", encoding='utf-8')
    test_df = pd.read_csv(DATA_DIR / "gsm8k_test.csv", encoding='utf-8')
    
    mid_point = len(test_df) // 2
    validation_df = test_df.iloc[:mid_point]
    test_df = test_df.iloc[mid_point:]
    
    train_df = train_df.sample(min(len(train_df), train), random_state=seed)
    validation_df = validation_df.sample(min(len(validation_df), dev), random_state=seed)
    test_df = test_df.sample(min(len(test_df), test), random_state=seed)
    
    return train_df, validation_df, test_df


def load_hot_qa(size=50, seed=42):
    
    with open(DATA_DIR / "hotpot.jsonl", encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    rng = random.Random(seed)
    
    return rng.sample(rows, k = min(size, len(rows)))

    
    
    