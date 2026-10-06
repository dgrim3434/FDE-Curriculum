from extraction.extractor import extract
from data.loaders import load_conll04 
from extraction.schemas import Graph
from llm import OllamaClient
MODELS = ['llama3.2:3b', 'llama3.2:1b', 'qwen2.5:3b']

def run_calibration():
    
    data = load_conll04(n=50)
    llm = OllamaClient('qwen2.5:3b')
    
    valid_schema = 0
    
    for idx in range(len(data)):
        row = data[idx]
        
        id = row['id']
        text = row['sentence']
        
        result = extract(text, Graph, llm, max_retries = 0, item_id=id)

        if result.outcome == 'first_pass':
            valid_schema += 1
    
    print(valid_schema)
    
if __name__ == "__main__":
    run_calibration()

# 'llama3.2:3b': first pass valid 16/ 50 = .32
# ''qwen2.5:3b': first pass valid 25/ 50 = .50