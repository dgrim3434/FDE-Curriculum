import re
from dataclasses import dataclass

from prompting.llm import LLMResponse


@dataclass(frozen=True)
class ClassificationResult:
    label: str | None
    valid: bool
    raw: LLMResponse


def label_key(s: str) -> str:
    """'Reverted card payment?' / '<answer>reverted_card_payment?</answer>' -> 'reverted_card_payment'"""
    s = re.sub(r"<[^>]+>", " ", s)                     # drop any XML-ish tags the model echoed
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def classify_intent(llm, system_prompt: str, user_prompt: str, valid_labels: set[str],
                    max_tokens: int = 30) -> ClassificationResult:
    resp = llm.complete(system=system_prompt, messages=[{"role": "user", "content": user_prompt}],
                        temperature=0.0, max_tokens=max_tokens)

    if resp.stop_reason != "end_turn":
        return ClassificationResult(label=None, valid=False, raw=resp)

    canonical = {label_key(l): l for l in valid_labels}
    label = canonical.get(label_key(resp.text))
    return ClassificationResult(label=label, valid=label is not None, raw=resp)