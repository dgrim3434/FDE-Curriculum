"""LLM Class converts the model response into a simple dataclass which is fully owned making testing and switching models much easier"""
from dataclasses import dataclass
import copy
import logging
import random
import time
import anthropic
import httpx
import ollama
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
llama_MODEL = "qwen2.5:3b"
HOST = "http://localhost:11434"

ANTH_MODEL = "claude-haiku-4-5-20251001"

ANTH_RETRIABLE = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.OverloadedError)
# ConnectionError = Ollama not running / unreachable. httpx.TimeoutException = request took too long.
LLAMA_RETRIABLE = (ConnectionError, httpx.TimeoutException)

# Ollama's done_reason -> the same stop_reason words used last week
STOP_REASONS = {"length": "max_tokens", "stop": "end_turn"}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float = 0.0
    model: str = llama_MODEL


class OllamaClient:

    def __init__(self, model: str = llama_MODEL, timeout: float = 120.0, max_attempts: int = 4, num_ctx: int = 8192):
        self.model = model
        self.max_attempts = max_attempts
        self.num_ctx = num_ctx
        self._client = ollama.Client(host=HOST, timeout=timeout)

    def complete(self, system: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512, seed=None, format=None) -> LLMResponse:

        full_messages = ([{"role": "system", "content": system}] if system else []) + messages

        options = dict(temperature=temperature, num_predict=max_tokens, num_ctx=self.num_ctx)
        
        
        if seed is not None:
            options['seed'] = seed
            
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.chat(model=self.model, messages=full_messages, options=options, format=format)
                break
            except LLAMA_RETRIABLE as e:

                if attempt == self.max_attempts:
                    raise

                wait = min(2 ** attempt, 30) + random.uniform(0, 1)
                logger.warning("attempt %d/%d failed (%s); retrying in %.1fs",
                               attempt, self.max_attempts, type(e).__name__, wait)

                time.sleep(wait)

        latency = time.perf_counter() - t0
        return LLMResponse(text=r.message.content,
                           stop_reason=STOP_REASONS.get(r.done_reason, r.done_reason),
                           input_tokens=r.prompt_eval_count or 0,
                           output_tokens=r.eval_count or 0,
                           latency_s=latency,
                           model=self.model)

    
        

class AnthropicClient:
    
    def __init__(self, model: str = ANTH_MODEL, timeout: float = 30.0, max_attempts: int = 4):
        load_dotenv()
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        
        self.model = model
        self.max_attempts = max_attempts
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=0)
    
    def complete(self, system: str, messages: list[dict], temperature = 0.0, max_tokens = 512, seed = None, format=None) -> LLMResponse:
        
        kwargs = dict(model = self.model, messages = messages, max_tokens = max_tokens, extra_body={"temperature": temperature})
        
        if system:
            kwargs['system'] = system
        
        for attempt in range(1, self.max_attempts + 1):
            
            try:
                r = self._client.messages.create(**kwargs)
                break
            except ANTH_RETRIABLE as e:
                
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

    
    def complete_with_tools(self, messages, tools, tool_choice=None, max_tokens=1024, temperature=0.0, system = None):
        
        kwargs = dict(model = self.model, messages = messages, max_tokens = max_tokens, extra_body={"temperature": temperature}, tools=tools)
        
        if system is not None:
            kwargs['system'] = system
        if tool_choice is not None:
            kwargs['tool_choice'] = tool_choice
        
        for attempt in range(1, self.max_attempts + 1):
            
            try:
                r = self._client.messages.create(**kwargs)
                break
            except ANTH_RETRIABLE as e:
                
                if attempt == self.max_attempts:
                    raise
                
                wait = min(2 ** attempt, 30) + random.uniform(0, 1)
                logger.warning("attempt %d/%d failed (%s); retrying in %.1fs",
                               attempt, self.max_attempts, type(e).__name__, wait)
                
                time.sleep(wait)
            
        return r
"""
Fake model response used for testing
"""

class FakeLLM:

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
                 stop: list[str] | None = None, seed: int | None = None, format=None) -> LLMResponse:
        # deepcopy: the caller keeps appending to the same `messages` list after this call returns
        self.calls.append({"system": system, "messages": copy.deepcopy(messages),
                           "temperature": temperature, "max_tokens": max_tokens,
                           "stop": stop, "seed": seed})
        if not self._responses:
            raise AssertionError(f"FakeLLM ran out of scripted responses on call {len(self.calls)}")
        return self._responses.pop(0)