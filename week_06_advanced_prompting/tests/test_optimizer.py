"""AI GENERATED TESTS"""
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from prompting.optimizer import (
    evaluate, propose, check_candidate, build_ngram, sign_test, paired_counts,
    log_version, optimizer, EvaluationResults, PromptCandidate, CandidateCheck,
)
from prompting.llm import FakeLLM, LLMResponse
from prompting.parsing import make_text_parser, make_number_parser, numbers_equal


# ─────────────────────────────── fixtures & fakes ──────────────────────

def H(prompt):
    return sha256(prompt.encode("utf-8")).hexdigest()[:12]


STORIES = {
    "train": "a farmer has {x} apples and buys more of them at the weekly market",
    "dev": "a baker bakes {x} loaves of bread every morning before the shop opens",
    "r": "a runner jogs {x} laps around the park track each evening after work",
}


def make_examples(n, offset=0, tag="train"):
    return [
        {"id": f"{tag}-{i}",
         "question": f"{tag} question {i}: " + STORIES[tag].format(x=i + offset),
         "gold": float(i + 1)}
        for i in range(n)
    ]


TRAIN = make_examples(10, tag="train")
DEV = make_examples(20, offset=100, tag="dev")

BASELINE = "Solve the problem. End with <answer>N</answer>."
MAGIC = "Solve it step by step MAGIC. End with <answer>N</answer>."
NOISE = "Solve it carefully NOISE. End with <answer>N</answer>."
WORSE = "Guess quickly WORSE. End with <answer>N</answer>."


def scorer(text, example):
    r = make_number_parser()(text)
    return SimpleNamespace(correct=bool(r.ok and numbers_equal(r.value, example["gold"])), parsed=r.ok)


class TaskFake:
    """Answers depend on the system prompt (positions counted inside each split):
       MAGIC → every question right; WORSE → none; NOISE → baseline plus ONE extra;
       anything else → the first half right."""

    def __init__(self, *splits):
        self.by_question = {}
        for examples in splits:
            for i, e in enumerate(examples):
                self.by_question[e["question"]] = (i, e["gold"], len(examples))
        self.calls = []

    def complete(self, system, messages, temperature=0.0, max_tokens=512, stop=None, seed=None):
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        q = messages[-1]["content"]
        idx, gold, n = self.by_question[q]
        half = n // 2
        if "MAGIC" in system:
            right = True
        elif "WORSE" in system:
            right = False
        elif "NOISE" in system:
            right = idx < half + 1
        else:
            right = idx < half
        value = gold if right else gold + 1
        return LLMResponse(text=f"work... <answer>{value:g}</answer>", stop_reason="end_turn",
                           input_tokens=10, output_tokens=5, latency_s=0.01)


def P(prompt, diagnosis="a diagnosis"):
    body = f"<diagnosis>{diagnosis}</diagnosis>" if diagnosis is not None else ""
    return LLMResponse(text=f"{body}<prompt>{prompt}</prompt>", stop_reason="end_turn",
                       input_tokens=10, output_tokens=5, latency_s=0.01)


META_SYSTEM = "You improve system prompts for a math model."


def metaprompt_builder(seen):
    def build(best_prompt, train_result, train):
        seen.append({"best": best_prompt, "train_result": train_result, "train": train})
        wrong = [e["question"] for e in train if not train_result.per_example[e["id"]]["correct"]]
        user = "CURRENT: " + best_prompt + "\nFAILURES:\n" + "\n".join(wrong)
        return META_SYSTEM, user
    return build


def banned():
    """Dev 8-grams, built at test time so one broken function can't block every test."""
    out = set()
    for e in DEV:
        out |= build_ngram(e["question"])
    return out


def run_opt(proposer_replies, tmp_path, rounds=1, k=1, alpha=0.1, patience=3, max_words=200):
    task = TaskFake(TRAIN, DEV)
    proposer = FakeLLM(proposer_replies)
    seen = []
    log_path = tmp_path / "versions.jsonl"
    best, reason = optimizer(
        task, proposer, BASELINE, TRAIN, DEV, scorer, metaprompt_builder(seen),
        make_text_parser("prompt"), make_text_parser("diagnosis"),
        ["<answer>"], max_words, banned(), rounds, k, alpha, patience, str(log_path),
    )
    lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return SimpleNamespace(best=best, reason=reason, log=lines, task=task, proposer=proposer, seen=seen)


# ─────────────────────────────── evaluate ──────────────────────────────

class TestEvaluate:
    def test_accuracy_and_per_example(self):
        res = evaluate(TaskFake(DEV), BASELINE, DEV, scorer, {})
        assert isinstance(res, EvaluationResults)
        assert res.n == 20
        assert res.accuracy == pytest.approx(0.5)
        assert isinstance(res.per_example, dict)
        assert set(res.per_example) == {e["id"] for e in DEV}
        assert res.per_example["dev-0"]["correct"] is True
        assert res.per_example["dev-19"]["correct"] is False

    def test_response_text_kept(self):
        res = evaluate(TaskFake(DEV), BASELINE, DEV, scorer, {})
        assert res.per_example["dev-0"]["response"] == "work... <answer>1</answer>"
        assert res.per_example["dev-19"]["response"] == "work... <answer>21</answer>"

    def test_response_text_survives_cache_hit(self):
        task, cache = TaskFake(DEV), {}
        evaluate(task, BASELINE, DEV, scorer, cache)
        again = evaluate(task, BASELINE, DEV, scorer, cache)
        assert again.per_example["dev-0"]["response"] == "work... <answer>1</answer>"

    def test_prompt_hash(self):
        assert evaluate(TaskFake(DEV), BASELINE, DEV, scorer, {}).prompt_hash == H(BASELINE)

    def test_messages_and_settings(self):
        task = TaskFake(DEV)
        evaluate(task, BASELINE, DEV[:3], scorer, {})
        assert len(task.calls) == 3
        for call, ex in zip(task.calls, DEV[:3]):
            assert call["system"] == BASELINE
            assert call["messages"][-1]["role"] == "user"
            assert call["messages"][-1]["content"] == ex["question"]
            assert call["temperature"] == 0.0

    def test_cache_prevents_repeat_calls(self):
        task, cache = TaskFake(DEV), {}
        first = evaluate(task, BASELINE, DEV, scorer, cache)
        n_calls = len(task.calls)
        second = evaluate(task, BASELINE, DEV, scorer, cache)
        assert len(task.calls) == n_calls
        assert second.per_example == first.per_example
        assert (H(BASELINE), "dev-0") in cache

    def test_cache_is_per_prompt(self):
        task, cache = TaskFake(DEV), {}
        evaluate(task, BASELINE, DEV, scorer, cache)
        evaluate(task, MAGIC, DEV, scorer, cache)
        assert len(task.calls) == 40

    def test_partial_cache_hit(self):
        task, cache = TaskFake(DEV), {}
        evaluate(task, BASELINE, DEV[:5], scorer, cache)
        evaluate(task, BASELINE, DEV, scorer, cache)
        assert len(task.calls) == 20

    def test_rates(self):
        replies = [
            LLMResponse("<answer>1</answer>", "end_turn", 10, 5, 0.1),
            LLMResponse("no tag here", "end_turn", 30, 5, 0.3),
            LLMResponse("<answer>9</answer> and more", "max_tokens", 20, 5, 0.2),
            LLMResponse("<answer>4</answer>", "end_turn", 40, 5, 0.4),
        ]
        exs = make_examples(4, tag="r")
        res = evaluate(FakeLLM(replies), BASELINE, exs, scorer, {})
        assert res.accuracy == pytest.approx(0.5)
        assert res.valid_parse_rate == pytest.approx(0.75)
        assert res.truncation_rate == pytest.approx(0.25)
        assert res.avg_input_tokens == pytest.approx(25.0)
        assert res.avg_latency == pytest.approx(0.25)


# ─────────────────────────────── propose ───────────────────────────────

class TestPropose:
    def run(self, replies, k):
        fake = FakeLLM(replies)
        cands, failures = propose(fake, "META SYSTEM", "METAPROMPT TEXT", k,
                                  make_text_parser("prompt"), make_text_parser("diagnosis"))
        return cands, failures, fake

    def test_k_calls_with_distinct_seeds_and_sampling(self):
        _, _, fake = self.run([P("A <answer>N</answer>"), P("B <answer>N</answer>"), P("C <answer>N</answer>")], 3)
        assert len(fake.calls) == 3
        assert len({c["seed"] for c in fake.calls}) == 3
        assert all(c["temperature"] > 0 for c in fake.calls)
        assert all(c["messages"][-1]["content"] == "METAPROMPT TEXT" for c in fake.calls)
        assert all(c["messages"][-1]["role"] == "user" for c in fake.calls)
        assert all(c["system"] == "META SYSTEM" for c in fake.calls)

    def test_candidates_parsed(self):
        cands, failures, _ = self.run([P("one two three <answer>N</answer>", "drops units")], 1)
        assert failures == 0
        assert len(cands) == 1
        c = cands[0]
        assert isinstance(c, PromptCandidate)
        assert c.prompt == "one two three <answer>N</answer>"
        assert c.diagnosis == "drops units"
        assert c.word_count == 4

    def test_missing_diagnosis_still_kept(self):
        cands, _, _ = self.run([P("only a prompt <answer>N</answer>", diagnosis=None)], 1)
        assert len(cands) == 1 and cands[0].diagnosis is None

    def test_unparseable_counts_as_failure(self):
        replies = [P("good <answer>N</answer>"), LLMResponse("no tags at all", "end_turn", 1, 1, 0.0)]
        cands, failures, _ = self.run(replies, 2)
        assert len(cands) == 1
        assert failures == 1

    def test_duplicates_dropped(self):
        cands, failures, _ = self.run([P("same <answer>N</answer>")] * 3, 3)
        assert len(cands) == 1
        assert failures == 0


# ─────────────────────────────── checks ────────────────────────────────

class TestBuildNgram:
    def test_count(self):
        assert len(build_ngram("one two three four five six seven eight nine")) == 2

    def test_short_text_is_empty_set(self):
        assert build_ngram("one two three four five") == set()

    def test_returns_strings(self):
        grams = build_ngram("one two three four five six seven eight")
        assert grams == {"one two three four five six seven eight"}

    def test_normalized(self):
        a = build_ngram("The Farmer, has SOME apples and buys more today")
        b = build_ngram("farmer has some apples and buys more today")
        assert a & b


class TestCheckCandidate:
    def check(self, text, seen=None, max_words=50):
        return check_candidate(text, ["<answer>"], max_words, banned(), seen or set())

    def test_ok(self):
        r = self.check("Solve step by step. End with <answer>N</answer>.")
        assert isinstance(r, CandidateCheck)
        assert r.ok is True and r.reason is None

    def test_duplicate(self):
        r = self.check(BASELINE, seen={H(BASELINE)})
        assert r.ok is False and r.reason

    def test_missing_required(self):
        r = self.check("Solve step by step and give the number.")
        assert r.ok is False and r.reason

    def test_too_long(self):
        r = self.check(" ".join(["word"] * 60) + " <answer>N</answer>", max_words=50)
        assert r.ok is False and r.reason

    def test_exactly_max_words_allowed(self):
        text = " ".join(["word"] * 9) + " <answer>N</answer>"
        assert self.check(text, max_words=10).ok is True

    def test_leakage(self):
        text = f"Remember this: {DEV[3]['question']}. End with <answer>N</answer>."
        r = self.check(text)
        assert r.ok is False and r.reason

    def test_train_text_is_not_leakage(self):
        text = f"Example: {TRAIN[3]['question']}. End with <answer>N</answer>."
        assert self.check(text, max_words=100).ok is True


# ─────────────────────────────── statistics ────────────────────────────

def PE(**correct):
    """per_example in evaluate's shape: PE(a=True) -> {"a": {"correct": True, "response": "..."}}"""
    return {k: {"correct": v, "response": f"<answer>{int(v)}</answer>"} for k, v in correct.items()}


class TestStats:
    @pytest.mark.parametrize("b,c,p", [(3, 9, 0.146), (9, 3, 0.146), (2, 12, 0.0129), (5, 5, 1.0), (0, 0, 1.0)])
    def test_sign_test(self, b, c, p):
        assert sign_test(b, c) == pytest.approx(p, abs=1e-3)

    def test_paired_counts(self):
        best = PE(a=True, b=True, c=False, d=False)
        cand = PE(a=True, b=False, c=True, d=False)
        assert paired_counts(best, cand) == (1, 1)

    def test_paired_counts_direction(self):
        assert paired_counts(PE(a=False, b=False), PE(a=True, b=True)) == (0, 2)
        assert paired_counts(PE(a=True, b=True), PE(a=False, b=False)) == (2, 0)

    def test_paired_counts_all_agree(self):
        assert paired_counts(PE(a=True, b=False), PE(a=True, b=False)) == (0, 0)

    def test_paired_counts_on_real_evaluate_output(self):
        task, cache = TaskFake(DEV), {}
        base = evaluate(task, BASELINE, DEV, scorer, cache)
        magic = evaluate(task, MAGIC, DEV, scorer, cache)
        assert paired_counts(base.per_example, magic.per_example) == (0, 10)

    def test_paired_counts_mismatched_ids(self):
        with pytest.raises(ValueError):
            paired_counts(PE(a=True), PE(b=True))


class TestLogVersion:
    def test_jsonl(self, tmp_path):
        path = tmp_path / "log.jsonl"
        log_version(str(path), {"version_id": 0, "status": "baseline"})
        log_version(str(path), {"version_id": 1, "status": "accepted"})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(l)["version_id"] for l in lines] == [0, 1]


# ─────────────────────────────── optimizer ─────────────────────────────

class TestOptimizer:
    def test_accepts_a_real_improvement(self, tmp_path):
        r = run_opt([P(MAGIC)], tmp_path)
        assert r.best == MAGIC
        accepted = [l for l in r.log if l["status"] == "accepted"]
        assert len(accepted) == 1
        assert accepted[0]["version_id"] == 1
        assert accepted[0]["parent_id"] == 0
        assert (accepted[0]["b"], accepted[0]["c"]) == (0, 10)

    def test_baseline_logged_first(self, tmp_path):
        r = run_opt([P(MAGIC)], tmp_path)
        base = r.log[0]
        assert base["version_id"] == 0 and base["status"] == "baseline"
        assert base["dev_acc"] == pytest.approx(0.5)
        assert base["train_acc"] == pytest.approx(0.5)

    def test_rejects_noise(self, tmp_path):
        r = run_opt([P(NOISE)], tmp_path)
        assert r.best == BASELINE
        rec = [l for l in r.log if l["version_id"] == 1][0]
        assert rec["status"] == "rejected_test"
        assert (rec["b"], rec["c"]) == (0, 1)

    def test_rejects_worse(self, tmp_path):
        r = run_opt([P(WORSE)], tmp_path)
        assert r.best == BASELINE
        assert [l for l in r.log if l["version_id"] == 1][0]["status"] == "rejected_test"

    def test_only_round_winner_is_tested(self, tmp_path):
        r = run_opt([P(NOISE), P(MAGIC)], tmp_path, k=2)
        assert r.best == MAGIC
        by_id = {l["version_id"]: l for l in r.log}
        assert by_id[1]["status"] == "rejected_test" and by_id[1]["p"] is None
        assert by_id[2]["status"] == "accepted"

    def test_failed_check_is_logged_and_never_scored(self, tmp_path):
        bad = "Solve it with no format instruction"
        r = run_opt([P(bad)], tmp_path)
        rec = [l for l in r.log if l["version_id"] == 1][0]
        assert rec["status"] == "rejected_check" and rec["reason"]
        assert rec["dev_acc"] is None
        assert not any(c["system"] == bad for c in r.task.calls)

    def test_leaking_candidate_rejected(self, tmp_path):
        leak = f"{DEV[0]['question']}. MAGIC. End with <answer>N</answer>."
        r = run_opt([P(leak)], tmp_path)
        assert r.best == BASELINE
        assert [l for l in r.log if l["version_id"] == 1][0]["status"] == "rejected_check"

    def test_duplicate_of_baseline_rejected(self, tmp_path):
        r = run_opt([P(BASELINE)], tmp_path)
        assert [l for l in r.log if l["version_id"] == 1][0]["status"] == "rejected_check"

    def test_every_candidate_logged_with_unique_ids(self, tmp_path):
        r = run_opt([P(NOISE), P(WORSE), P("no format here")], tmp_path, k=3)
        ids = [l["version_id"] for l in r.log]
        assert sorted(ids) == [0, 1, 2, 3]          # unique, none skipped (order may vary)
        assert len(r.log) == 4
        required = {"version_id", "parent_id", "round", "status", "reason", "prompt_text", "diagnosis",
                    "train_acc", "dev_acc", "b", "c", "p"}
        for line in r.log:
            assert required <= set(line)

    def test_stops_on_rounds(self, tmp_path):
        r = run_opt([P(NOISE)], tmp_path, rounds=1, patience=5)
        assert r.reason == "rounds"

    def test_stops_on_patience(self, tmp_path):
        replies = [P(f"Try variant {i} NOISE. End with <answer>N</answer>.") for i in range(10)]
        r = run_opt(replies, tmp_path, rounds=6, k=1, patience=2)
        assert r.reason == "patience"
        assert len(r.proposer.calls) == 2

    def test_empty_round_counts_toward_patience(self, tmp_path):
        replies = [LLMResponse("nothing useful", "end_turn", 1, 1, 0.0)] * 10
        r = run_opt(replies, tmp_path, rounds=6, k=1, patience=2)
        assert r.reason == "patience"
        assert len(r.proposer.calls) == 2

    def test_acceptance_resets_patience_and_updates_parent(self, tmp_path):
        replies = [P(MAGIC)] + [P(f"Variant {i} MAGIC also. End with <answer>N</answer>.") for i in range(5)]
        r = run_opt(replies, tmp_path, rounds=3, k=1, patience=2)
        round2 = [l for l in r.log if l["round"] == 2]
        assert round2 and all(l["parent_id"] == 1 for l in round2)

    def test_metaprompt_built_from_train_only(self, tmp_path):
        r = run_opt([P(NOISE)], tmp_path)
        assert r.seen and r.seen[0]["train"] is TRAIN
        sent = " ".join(m["content"] for c in r.proposer.calls for m in c["messages"])
        sent += " ".join(c["system"] for c in r.proposer.calls)
        assert not any(e["question"] in sent for e in DEV)

    def test_metaprompt_system_and_user_reach_proposer(self, tmp_path):
        r = run_opt([P(NOISE)], tmp_path)
        call = r.proposer.calls[0]
        assert call["system"] == META_SYSTEM
        assert call["messages"][-1]["content"].startswith("CURRENT: " + BASELINE)

    def test_metaprompt_receives_train_responses(self, tmp_path):
        r = run_opt([P(NOISE)], tmp_path)
        per = r.seen[0]["train_result"].per_example
        assert per["train-9"]["correct"] is False
        assert per["train-9"]["response"] == "work... <answer>11</answer>"

    def test_metaprompt_uses_current_best(self, tmp_path):
        replies = [P(MAGIC), P("Another one NOISE. End with <answer>N</answer>.")]
        r = run_opt(replies, tmp_path, rounds=2, k=1, patience=5)
        assert r.seen[0]["best"] == BASELINE
        assert r.seen[1]["best"] == MAGIC

    def test_best_never_rescored_needlessly(self, tmp_path):
        r = run_opt([P(NOISE)], tmp_path)
        baseline_dev_calls = [c for c in r.task.calls
                              if c["system"] == BASELINE and c["messages"][-1]["content"].startswith("dev")]
        assert len(baseline_dev_calls) == len(DEV)