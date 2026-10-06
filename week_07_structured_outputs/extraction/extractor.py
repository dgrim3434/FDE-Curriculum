import logging
from dataclasses import dataclass, field

from extraction.prompts import build_prompt
from extraction.repair import run_pipeline

logger = logging.getLogger(__name__)

MAX_TOKENS = 2048

@dataclass
class ExtractionResult:
    value: object | None
    outcome: str
    attempts: list = field(default_factory=list)
    raw_outputs: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0



def extract(text, schema, llm, max_retries=2, max_tokens=1024, item_id="", temperature=0, seed=0, format=None):
    
    sys, user = build_prompt(text, schema)
    
    messages = [{'role': 'user', 'content': user}]
    attempts, raw_output = [], []
    input_tokens, output_tokens = 0,0
    
    curr_tokens = max_tokens
    
    for i in range(max_retries + 1):
        
        resp = llm.complete(system = sys, messages=messages, temperature=temperature, max_tokens=curr_tokens, seed=seed, format=format)
        raw_output.append(resp.text)
        input_tokens += resp.input_tokens
        output_tokens += resp.output_tokens
        
        
        result = run_pipeline(resp.text, schema=schema, done_reason=resp.stop_reason)
        attempts.append(result)
        
        if result.ok:
            
            outcome = "first_pass"
            
            if len(result.rules_fired) > 0:
                outcome = "repaired"
            
            if i > 0:
                outcome = "retry"
            
            return ExtractionResult(value=result.value, outcome=outcome, attempts=attempts, raw_outputs=raw_output, input_tokens=input_tokens, output_tokens=output_tokens)
        
        else:
            if result.failed_stage == "truncated":
                
                if curr_tokens == MAX_TOKENS:
                    return ExtractionResult(value=None, outcome="failed", attempts=attempts, raw_outputs=raw_output, input_tokens=input_tokens, output_tokens=output_tokens)
                
                logger.warning("[%s] attempt %d truncated, raising max_tokens to %d",
                               item_id, i + 1, min(int(curr_tokens * 1.5), MAX_TOKENS))
                
                curr_tokens = min(int(curr_tokens * 1.5), MAX_TOKENS)

            else:
                
                logger.warning("[%s] schema=%s attempt=%d/%d FAILED %s | rules fired: %s",
                               item_id, schema.__name__, i + 1, max_retries + 1,
                               result.failed_stage, result.rules_fired)
                               
                messages.append({'role': 'assistant', 'content': resp.text or "(empty response)"})
                messages.append({'role': 'user', 'content': result.model_message})

    return ExtractionResult(value=None, outcome="failed", attempts=attempts, raw_outputs=raw_output, input_tokens=input_tokens, output_tokens=output_tokens)


if __name__ == "__main__":
    
    from llm import OllamaClient, AnthropicClient
    from extraction.schemas import Extraction, Event
    sentence = "EU rejects German call to boycott British lamb ."
    
    for llm in [OllamaClient(), AnthropicClient()]:
        
        r = extract(sentence, Extraction, llm, item_id="test_0")
        print(llm.model, r.outcome, len(r.attempts), r.value, r.input_tokens, r.output_tokens)