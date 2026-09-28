"""
Run from week_05_prompting/:
    python -m experiments.CoT.run_cot direct
    python -m experiments.CoT.run_cot cot
    python -m experiments.CoT.run_cot cot_3shot
"""
import json
import sys

import pandas as pd

from experiments.CoT.constants import (ANSWER_COLUMN, GOLDEN_COLUMN, MAX_TOKENS, QUESTION_COLUMN,
                                       RESULTS_DIR, SEED, compute_result, compute_summary)
from experiments.load_data import load_gsm8k
from prompting.budget import BudgetTracker
from prompting.chain_of_thought import chain_of_thought
from prompting.llm import AnthropicClient
from prompting.templates import render

CONFIGS = {
    "direct":     (False,   0,          "user/math_problem_no_COT.j2"),
    "cot":        (True,    0,          "user/math_solver_CoT.j2"),
    "cot_3shot":  (True,    3,          "user/math_solver_CoT.j2"),
}


def make_examples(train: pd.DataFrame, n: int) -> list[dict]:
    rows = train.head(n)
    return [{"question": r[QUESTION_COLUMN],
             "thinking": r[ANSWER_COLUMN].split("####")[0].replace("<<", " ").replace(">>", " ").strip(),
             "answer": r[GOLDEN_COLUMN]}
            for _, r in rows.iterrows()]


def run(name: str, n: int = 50, budget_usd: float = 1.0):
    use_cot, n_examples, user_tpl = CONFIGS[name]
    llm = AnthropicClient()                       # created when the run starts, not at import time
    train, test = load_gsm8k(testing_size=n, seed=SEED)
    examples = make_examples(train, n_examples) if n_examples else None
    system_prompt = render("system/math_solver.j2")
    budget = BudgetTracker(budget_usd)

    csv_path = RESULTS_DIR / f"gsm8k_{name}.csv"
    rows = []
    try:
        for idx, row in enumerate(test.itertuples(index=False)):
            question, gold = getattr(row, QUESTION_COLUMN), getattr(row, GOLDEN_COLUMN)
            resp = chain_of_thought(llm, system_prompt, render(user_tpl, query=question), budget,
                                    examples=examples, max_tokens=MAX_TOKENS, use_cot=use_cot)
            rows.append(compute_result(resp, idx, question, gold))
            if resp.reason == "budget":
                print("budget reached, stopping early")
                break
    finally:                                      # a crash on row 49 still saves rows 0-48
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = compute_summary(df, method=name, examples=n_examples, model=llm.model)
    (RESULTS_DIR / f"gsm8k_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "cot")