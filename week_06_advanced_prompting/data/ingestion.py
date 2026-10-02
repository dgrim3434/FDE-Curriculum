from datasets import load_dataset
import pandas as pd
from data.constants import DATA_DIR

"""
def save_gsm8k():
    
    ds = load_dataset("openai/gsm8k", "main")
    print(ds)
    
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        
        df['gold'] = df['answer'].str.split("####").str[-1].str.strip().str.replace(",", "").astype(float)
        df.to_csv(DATA_DIR / f"gsm8k_{split}.csv", index=False, encoding="utf-8")
        print(f"gsm8k {split}: {len(df)} rows -> {df.columns.to_list()}")

"""

def ingest_data():
    
    try:
        gsm = load_dataset("openai/gsm8k", "main")
        
        for split in ("train", "test"):
            df = gsm[split].to_pandas()
            
            df['gold'] = df['answer'].str.split("####").str[-1].str.strip().str.replace(",", "").astype(float)
            df.to_csv(DATA_DIR / f"gsm8k_{split}.csv", index=False, encoding='utf-8')
            
    except Exception as e:
        print(f"Unable to load GSM8K dataset from huggingface. Recieved error: {e}")
    
    try:
        hot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        hot.to_json(DATA_DIR / "hotpot.jsonl")
    except Exception as e:
        print(f"Unable to load hotpotqa dataset from huggingface. Recieved error: {e}")


if __name__ == "__main__":
    
    ingest_data()
    
