from extraction.schemas import EntityTypes, Extraction, Event, Graph
from data.loaders import load_conll04, load_conll2003
from llm import AnthropicClient, OllamaClient
import json
from pathlib import Path
from extraction.extractor import extract

RESULTS = Path(__file__).resolve().parent.parent / "results"


SCHEMAS = [EntityTypes, Extraction, Event, Graph]

llama_llm = OllamaClient()
claude_llm = AnthropicClient()

MODELS = [llama_llm, claude_llm]

def run_chart():
    
    data = load_conll2003(n=50)
    g_data = load_conll04(n=100)[50:]
    
    for llm in MODELS:
        
        model_results = []
        
        for schema in SCHEMAS:
            
            curr_data = data
            
            if schema is Graph:
                curr_data = g_data
            
            for idx in range(len(curr_data)):
                row = curr_data[idx]
                
                id = row['id']
                text = row['sentence']
                
                resp = extract(text=text, schema=schema, llm=llm, max_retries=2, item_id=id)
                
                record = {'id': id, 'tier': llm.model, 'schema': schema.__name__, "schema_version": 1, "outcome": resp.outcome, 'attempt_stages': [a.failed_stage for a in resp.attempts],
                          'rules_fired': [a.rules_fired for a in resp.attempts], 'raw_outputs': resp.raw_outputs, 'value': resp.value.model_dump() if resp.value else None,
                          'gold': row, 'input_tokens': resp.input_tokens, 'output_tokens': resp.output_tokens}
                
                model_results.append(record)

        safe = llm.model.replace(":", "_").replace(".", "-")
        out = RESULTS / f"run_{safe}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(model_results, indent=2), encoding='utf-8')


if __name__ == "__main__":
    run_chart()