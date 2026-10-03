# FDE Curriculum

Building the LLM engineering stack from first principles — tokenizers to transformers to agents — as a 66-week Forward Deployed Engineer curriculum. One folder per week; every week ships tested code, measured results, and a written analysis.

## Highlights

- **Prompt optimizer with a statistical acceptance gate** — train/dev/test separation, test-set leakage checks, and a sign test before any prompt is accepted. Raised GSM8K accuracy by **53 points** on a held-out test set using a local 3B model (p ≈ 2.5e-32; test set used once).
- **Measured the cost of orchestration on small models** — on HotpotQA, a single call beat a 3-step prompt chain (**62% vs. 46%**) on Llama 3.2 3B.
- **Provider-agnostic LLM interface** — the same pipelines run on Claude or on local models via Ollama (Llama 3.2 3B, Qwen 2.5 7B on an RTX 2080 Ti) with no code changes.
- **248 automated tests**, including scripted-LLM probes of error and retry paths.

## Progress

| Week | Topic | What was built | Status |
| --- | --- | --- | --- |
| [01](week_01_tokenization/) | Tokenization | BPE tokenizer from scratch: training, encode/decode, save/load | Complete |
| [02](week_02_embedding/) | Embeddings | Word2Vec skip-gram with negative sampling, sinusoidal positional encoding | Complete |
| [03](week_03_attention/) | Attention | Single- and multi-head self-attention, causal masking | Complete |
| [04](week_04_transformer/) | Transformer Architecture | Decoder-only transformer with configurable depth, trained end to end | Complete |
| [05](week_05_prompting/) | Prompting Foundations | Template library, embedding-based few-shot selector, structured-output parser with retry, chain-of-thought wrapper | Complete |
| [06](week_06_advanced_prompting/) | Advanced Prompting | Prompt chaining with branching, self-consistency with confidence scoring, ReAct agent with tools, prompt optimizer | Complete |
| 07 | Structured Outputs | Pydantic extraction, validation and repair, Claude tool-use formatting | In progress |

## Running

Requires **Python 3.12+**. Run the test suite with:

    python -m pytest