"""
Ingests the banking77 and the GSM8K from Hugging Face and saves them as CSVs within the data folder. This data will be used within the experients
"""


from pathlib import Path
from datasets import load_dataset
import pandas as pd
DATA_DIR = Path(__file__).resolve().parent

BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"


def save_banking77():
    for split in ("train", "test"):
        df = pd.read_csv(BANKING77_URL.format(split=split), encoding="utf-8")
        df = df.rename(columns={"category": "label_name"})       # intent name, e.g. "card_arrival"
        df.to_csv(DATA_DIR / f"banking77_{split}.csv", index=False, encoding="utf-8")
        print(f"banking77 {split}: {len(df)} rows, {df['label_name'].nunique()} intents")

def save_gsm8k():
    
    ds = load_dataset("openai/gsm8k", "main")
    print(ds)
    
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        
        df['gold'] = df['answer'].str.split("####").str[-1].str.strip().str.replace(",", "").astype(float)
        df.to_csv(DATA_DIR / f"gsm8k_{split}.csv", index=False, encoding="utf-8")
        print(f"gsm8k {split}: {len(df)} rows -> {df.columns.to_list()}")

if __name__ == "__main__":
    
    save_banking77()
    save_gsm8k()