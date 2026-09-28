"""LLM Class converts the model response into a simple dataclass which is fully owned making testing and switching models much easier"""
from dataclasses import dataclass
import anthropic
from dotenv import load_dotenv
from prompting.cost import call_price
import time, random
MODEL = "claude-haiku-4-5-20251001"

RETRIABLE = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)

@dataclass(frozen=True)
class LLMResponse:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str = MODEL
    
    @property
    def cost(self) -> float:
        return call_price(self.model, self.input_tokens, self.output_tokens)

class AnthropicClient:
    
    def __init__(self, model: str = MODEL, timeout: float = 30.0):
        load_dotenv()
        self.model = model
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=0)
    
    def complete(self, system: str, messages: list[dict], temperature = 0.0, max_tokens = 512, max_attempts=4) -> LLMResponse:
        
        kwargs = dict(model = self.model, messages = messages, max_tokens = max_tokens, extra_body={"temperature": temperature})
        
        if system:
            kwargs['system'] = system
        
        r = self._client.messages.create(**kwargs)
        
        text = "".join(block.text for block in r.content if block.type == "text")
        
        attempt = 1
        while attempt <= max_attempts:
            try:
                return LLMResponse(text=text, stop_reason=r.stop_reason, input_tokens= r.usage.input_tokens, output_tokens=r.usage.output_tokens)
            except RETRIABLE:
                wait = 2 ** attempt + random.uniform(0, 1)
                attempt += 1
                time.sleep(wait)

"""
Fake model response used for testing
"""

class FakeLLM:
    
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []
    
    def complete(self, system: str, messages: list[dict], temperature=0.0, max_tokens = 512) -> LLMResponse:
        
        self.calls.append({"system": system, "messages": messages, temperature: temperature, "max_tokens": max_tokens})
        
        return self._responses.pop(0)
