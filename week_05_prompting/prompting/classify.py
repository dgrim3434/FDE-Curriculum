from dataclasses import dataclass
from prompting.llm import LLMResponse

@dataclass(frozen = True)
class ClassificationResult:
    label: str | None 
    valid: bool
    raw: LLMResponse


def classify_intent(llm, system_prompt: str, user_prompt: str, valid_labels: set[str]) -> ClassificationResult:
    
    resp = llm.complete(system = system_prompt, messages = [{'role': 'user', 'content': user_prompt}], temperature=0.0, max_tokens=20)
    
    if resp.stop_reason != "end_turn":
        return ClassificationResult(label=None, valid=False, raw=resp)
    
    label = resp.text.strip().lower().replace(" ", "_")
    
    canonical = {l.lower(): l for l in valid_labels}
    
    if label in canonical:
        label = canonical.get(label)
        return ClassificationResult(label=label, valid=True, raw=resp)
    
    return ClassificationResult(label=None, valid=False, raw=resp)