from pathlib import Path

from prompting.chain_of_thought import CoTResult

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "cot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUESTION_COLUMN = "question"
ANSWER_COLUMN = "answer"
GOLDEN_COLUMN = "gold"
MAX_TOKENS = 512
SEED = 42


def compute_result(resp: CoTResult, idx, question, gold) -> dict:
    correct = resp.valid and abs(resp.answer - gold) < 1e-6
    return {
        "idx": idx, "question": question, "gold": gold,
        "pred": resp.answer, "correct": bool(correct),
        "valid": resp.valid, "reason": resp.reason, "method": resp.method,
        "format_ok": resp.format_ok, "attempts": resp.attempts,
        "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
        "cost": resp.total_cost,
        "thought": resp.thought.replace("\n", " ") if resp.thought else None,
        "error_log": " | ".join(resp.error_log),
    }


def _rate(num, den):
    return float(num / den) if den else None


def compute_summary(df, method: str, examples: int, model: str) -> dict:
    n, n_valid, n_correct = len(df), int(df["valid"].sum()), int(df["correct"].sum())
    total_cost = float(df["cost"].sum())
    return {
        "method": method, "model": model, "n": n, "seed": SEED, "max_tokens": MAX_TOKENS, "examples": examples,
        "accuracy": _rate(n_correct, n),
        "valid_rate": _rate(n_valid, n),
        "accuracy_when_valid": _rate(n_correct, n_valid),
        "format_ok_rate": _rate(int(df["format_ok"].sum()), n),
        "average_attempts": float(df["attempts"].mean()),
        "avg_output_tokens": float(df["output_tokens"].mean()),
        "total_cost": total_cost,
        "cost_per_correct": _rate(total_cost, n_correct),
        "reasons": df["reason"].value_counts().to_dict(),
        "methods": {str(k): int(v) for k, v in df["method"].fillna("none").value_counts().items()},
    }