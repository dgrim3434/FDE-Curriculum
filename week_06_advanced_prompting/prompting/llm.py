"""LLM Class converts the model response into a simple dataclass which is fully owned making testing and switching models much easier"""
from dataclasses import dataclass
import copy
import logging
import random
import time

import httpx
import ollama

logger = logging.getLogger(__name__)
MODEL = "llama3.2:3b"
HOST = "http://localhost:11434"

# ConnectionError = Ollama not running / unreachable. httpx.TimeoutException = request took too long.
RETRIABLE = (ConnectionError, httpx.TimeoutException)

# Ollama's done_reason -> the same stop_reason words used last week
STOP_REASONS = {"length": "max_tokens", "stop": "end_turn"}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float = 0.0
    model: str = MODEL


class OllamaClient:

    def __init__(self, model: str = MODEL, timeout: float = 120.0, max_attempts: int = 4, num_ctx: int = 8192):
        self.model = model
        self.max_attempts = max_attempts
        self.num_ctx = num_ctx
        self._client = ollama.Client(host=HOST, timeout=timeout)

    def complete(self, system: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
                 stop: list[str] | None = None, seed: int | None = None) -> LLMResponse:

        full_messages = ([{"role": "system", "content": system}] if system else []) + messages

        options = dict(temperature=temperature, num_predict=max_tokens, num_ctx=self.num_ctx)
        if stop:
            options["stop"] = stop
        if seed is not None:
            options["seed"] = seed

        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.chat(model=self.model, messages=full_messages, options=options)
                break
            except RETRIABLE as e:

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


"""
Fake model response used for testing
"""

class FakeLLM:

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
                 stop: list[str] | None = None, seed: int | None = None) -> LLMResponse:
        # deepcopy: the caller keeps appending to the same `messages` list after this call returns
        self.calls.append({"system": system, "messages": copy.deepcopy(messages),
                           "temperature": temperature, "max_tokens": max_tokens,
                           "stop": stop, "seed": seed})
        if not self._responses:
            raise AssertionError(f"FakeLLM ran out of scripted responses on call {len(self.calls)}")
        return self._responses.pop(0)