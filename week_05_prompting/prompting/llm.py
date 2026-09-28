"""LLM Class converts the model response into a simple dataclass which is fully owned making testing and switching models much easier"""
from dataclasses import dataclass
import anthropic
from dotenv import load_dotenv
from prompting.cost import call_price
import time, random
import logging
import os


logger = logging.getLogger(__name__)
MODEL = "claude-haiku-4-5-20251001"

RETRIABLE = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.OverloadedError)

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
    
    def __init__(self, model: str = MODEL, timeout: float = 30.0, max_attempts: int = 4):
        load_dotenv()
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        
        self.model = model
        self.max_attempts = max_attempts
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=0)
    
    def complete(self, system: str, messages: list[dict], temperature = 0.0, max_tokens = 512) -> LLMResponse:
        
        kwargs = dict(model = self.model, messages = messages, max_tokens = max_tokens, extra_body={"temperature": temperature})
        
        if system:
            kwargs['system'] = system
        
        for attempt in range(1, self.max_attempts + 1):
            
            try:
                r = self._client.messages.create(**kwargs)
                break
            except RETRIABLE as e:
                
                if attempt == self.max_attempts:
                    raise
                
                wait = min(2 ** attempt, 30) + random.uniform(0, 1)
                logger.warning("attempt %d/%d failed (%s); retrying in %.1fs",
                               attempt, self.max_attempts, type(e).__name__, wait)
                
                time.sleep(wait)
        
        text = "".join(block.text for block in r.content if block.type == "text")
        return LLMResponse(text=text, stop_reason=r.stop_reason,
                           input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                           model=self.model)
    
        

"""
Fake model response used for testing
"""

class FakeLLM:
    """Scripted stand-in for AnthropicClient in tests: returns `responses` in order, records every call."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 512) -> LLMResponse:
        # deepcopy: the caller keeps appending to the same `messages` list after this call returns
        self.calls.append({"system": system, "messages": copy.deepcopy(messages),
                           "temperature": temperature, "max_tokens": max_tokens})
        if not self._responses:
            raise AssertionError(f"FakeLLM ran out of scripted responses on call {len(self.calls)}")
        return self._responses.pop(0)
