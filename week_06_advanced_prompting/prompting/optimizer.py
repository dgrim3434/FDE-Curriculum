from prompting.templates import render
from prompting.parsing import make_text_parser, make_number_parser, numbers_equal, normalize_answer
from dataclasses import dataclass
from hashlib import sha256
import math
import json
from demos.constants import RESULTS_DIR

@dataclass
class EvaluationResults:
    prompt_hash: str
    per_example: list
    n : int
    accuracy: float
    truncation_rate: float
    avg_input_tokens: float
    valid_parse_rate: float
    avg_latency: float

def get_prompt_hash(text):
    
    return sha256(text.encode('utf-8')).hexdigest()[:12]

def evaluate(llm, system_prompt, examples, scorer, cache):
    
    prompt_hash = get_prompt_hash(system_prompt)
    total = 0
    correct = 0
    trun = 0
    latency = 0
    input_tok = 0
    parsed = 0
    per_example = {}
    for ex in examples:
        
        id = ex['id']
        example = ex['question']
        
        result = cache.get((prompt_hash, id))
        
        if result is None:
            user_prompt = [{'role': 'user', 'content': example}]
        
            resp = llm.complete(system=system_prompt, messages=user_prompt, temperature=0.0, max_tokens=1024, seed=0)
            
            truncated = resp.stop_reason == 'max_tokens'
            score = scorer(resp.text, ex)
            
            result = {'correct': score.correct, 'text': resp.text, 'parsed_ok': score.parsed, 'truncated': truncated, 'input_tokens': resp.input_tokens, 'output_tokens': resp.output_tokens, 'latency': resp.latency_s}
            cache[(prompt_hash, id)] = result
        
        total += 1
        if result['correct']:
            correct += 1
        if result['truncated']:
            trun += 1
        if result['parsed_ok']:
            parsed += 1
            
        latency += result['latency']
        input_tok += result['input_tokens']
        per_example[id] = {'correct': result['correct'], 'response': result['text']}
        
    return EvaluationResults(prompt_hash=prompt_hash, n=total, per_example = per_example, accuracy=float(correct / total), truncation_rate=float(trun / total), avg_input_tokens= (input_tok / total),
                             valid_parse_rate= (parsed / total), avg_latency= (latency / total))
    
@dataclass
class PromptCandidate:
    prompt: str
    diagnosis: str
    word_count: int          

def propose(proposer_llm, system, user, k, prompt_parser, diagnosis_parser):
    
    user_prompt = [{'role': 'user', 'content': user}]
    seen = set()
    candidates = []
    failures = 0
    base_seed = 0
    duplicates = 0
    for i in range(k):
        
        
        resp = proposer_llm.complete(system=system,messages=user_prompt, temperature=0.8, max_tokens=1024, seed=base_seed + i)

        prompt = prompt_parser(resp.text)
        diag = diagnosis_parser(resp.text)

        if prompt.ok and prompt.value not in seen:
            seen.add(prompt.value)
            diagnosis = None
            if diag.ok:
                diagnosis = diag.value
            
            candidates.append(PromptCandidate(prompt=prompt.value, diagnosis=diagnosis, word_count=len(prompt.value.split())))
        else:
            if not prompt.ok:
                failures += 1
            
            duplicates += 1
    
    return candidates, failures
    

@dataclass
class CandidateCheck:
    ok: bool
    reason: str | None

def check_candidate(text, required, max_words, banned_ngrams, seen_hashes):
    
    hash = get_prompt_hash(text)
    
    if hash in seen_hashes:
        return CandidateCheck(ok=False, reason="Duplicate prompt detected")
    
    for req in required:
            if req not in text:
                return CandidateCheck(ok=False, reason=f"Prompt missing required word: {req}. The prompt must contain all required words: {required}")
    
    if len(text.split()) > max_words:
        return CandidateCheck(ok=False, reason=f"Prompt exceeded the max_word count. Prompt Length was {len(text.split())}. Max length set to {max_words}")
    
    
    
    grams = set(build_ngram(text))
    
    if len(grams & banned_ngrams) > 0:
        return CandidateCheck(ok=False, reason= f"Found invalid sequence within the prompt. Sentence matched the banned list: {(grams & banned_ngrams).pop()}")
    
    return CandidateCheck(ok=True, reason=None)
    
    
def build_ngram(text, n = 8):
    
    norm_text = normalize_answer(text)
    words = norm_text.split()
    
    grams = set()
    
    for i in range(len(words) - n + 1):
        
        grams.add(" ".join(words[i:i + n]))
    
    return grams
    
    

def sign_test(b, c):
    
    denom = b + c
    
    val = max(b,c)
    num = sum([math.comb(denom, i) for i in range(val, denom + 1)])
    den = 2 ** denom
    
    p = num / den
    
    return min(1, 2 * p)

def paired_counts(best, current):
    
    if best.keys() != current.keys():
        raise ValueError("Prompts where scored on different questions.")
    
    b = 0
    c = 0
    for key in best.keys():
        
        b_ex = best[key]
        c_ex = current[key]
        
        if b_ex['correct'] and not c_ex['correct']:
            
            b += 1
        if c_ex['correct'] and not b_ex['correct']:
            c +=1
    
    return b, c
    
def optimizer(task_llm, proposer_llm, baseline_prompt, train, dev, scorer, build_metaprompt, prompt_parser, diagnosis_parser, required, max_words, banned_ngrams, rounds, k, alpha, patience, log_path):
    
    cache = {}
    seen_hashes = set()

    base_train = evaluate(llm=task_llm, system_prompt=baseline_prompt, examples=train, scorer=scorer, cache=cache)
    base_dev = evaluate(llm=task_llm, system_prompt=baseline_prompt, examples=dev, scorer=scorer, cache=cache)
    
    log = {'version_id': 0, 'parent_id': None, 'round': 0, 'status': 'baseline', 'reason': None, 'prompt_text': baseline_prompt, 'diagnosis': None, 'train_acc': base_train.accuracy,
           'dev_acc': base_dev.accuracy, 'b': None, 'c': None, 'p': None, 'format_pass_rate': base_dev.valid_parse_rate, 'avg_input_tokens': base_dev.avg_input_tokens, 'sec_per_question': base_dev.avg_latency}
    
    log_version(path=log_path, record=log)
    
    seen_hashes.add(get_prompt_hash(baseline_prompt))
    
    best = baseline_prompt
    
    curr_id = 0
    best_id = 0
    rounds_since_update = 0
    for round in range(rounds):
        
        eval = evaluate(llm=task_llm, system_prompt=best, examples=train, scorer=scorer, cache=cache)
        
        system, user = build_metaprompt(best, eval, train)
        
        candidates, n_failed = propose(proposer_llm, system, user, k, prompt_parser, diagnosis_parser)
        
        curr_best = {'train_acc': None, 'dev': None, 'prompt': None, 'diagnosis': None, 'id': None}
        
        for candidate in candidates:
            curr_id += 1
            check = check_candidate(candidate.prompt, required, max_words, banned_ngrams, seen_hashes)
            
            if not check.ok:
                
                log = {'version_id': curr_id, 'parent_id': best_id, 'round': round + 1, 'status': 'rejected_check', 'reason': check.reason, 'prompt_text': candidate.prompt, 'diagnosis': candidate.diagnosis,
                       'train_acc': None, 'dev_acc': None, 'b': None, 'c': None, 'p': None, 'format_pass_rate': None, 'avg_input_tokens': None, 'sec_per_question': None}
                log_version(path=log_path, record=log)
                continue
            
            seen_hashes.add(get_prompt_hash(candidate.prompt))
            
            train_res = evaluate(llm=task_llm, system_prompt=candidate.prompt, examples=train, scorer=scorer, cache=cache)
            dev_res = evaluate(llm=task_llm, system_prompt=candidate.prompt, examples=dev, scorer=scorer, cache=cache)

            if curr_best['id'] is None or dev_res.accuracy > curr_best['dev'].accuracy:
                
                if curr_best['id'] is not None:
                    log = {'version_id': curr_best['id'], 'parent_id': best_id, 'round': round + 1, 'status': 'rejected_test', 'reason': 'not round winnder', 'prompt_text': curr_best['prompt'],
                           'diagnosis': curr_best['diagnosis'], 'train_acc': curr_best['train_acc'], 'dev_acc': curr_best['dev'].accuracy, 'b': None, 'c': None, 'p': None,
                           'format_pass_rate': curr_best['dev'].valid_parse_rate, 'avg_input_tokens': curr_best['dev'].avg_input_tokens, 'sec_per_question': curr_best['dev'].avg_latency
                           }
                    log_version(path=log_path, record=log)
                curr_best['train_acc'] = train_res.accuracy
                curr_best['dev'] = dev_res
                curr_best['prompt'] = candidate.prompt
                curr_best['diagnosis'] = candidate.diagnosis
                curr_best['id'] = curr_id
            else:
                log = {'version_id': curr_id, 'parent_id': best_id, 'round': round + 1, 'status': 'rejected_test', 'reason': 'not round winnder', 'prompt_text': candidate.prompt,
                        'diagnosis': candidate.diagnosis, 'train_acc': train_res.accuracy, 'dev_acc': dev_res.accuracy, 'b': None, 'c': None, 'p': None,
                        'format_pass_rate': dev_res.valid_parse_rate, 'avg_input_tokens': dev_res.avg_input_tokens, 'sec_per_question': dev_res.avg_latency
                        }
                log_version(path=log_path, record=log)
        
        if curr_best['id'] is None:
            rounds_since_update += 1
            
            if rounds_since_update >= patience:
                return best, "patience"
            continue
        
        best_dev = evaluate(llm=task_llm, system_prompt=best, examples=dev, scorer=scorer, cache=cache)
        curr_dev = evaluate(llm=task_llm, system_prompt=curr_best['prompt'], examples=dev, scorer=scorer, cache=cache)
        
        b, c = paired_counts(best_dev.per_example, curr_dev.per_example)
        p = sign_test(b, c)
        if c > b and  p <= alpha:
            rounds_since_update = 0
            log = {'version_id': curr_best['id'], 'parent_id': best_id, 'round': round + 1, 'status': 'accepted', 'reason': None, 'prompt_text': curr_best['prompt'], 'diagnosis': curr_best['diagnosis'],
                   'train_acc': curr_best['train_acc'], 'dev_acc': curr_best['dev'].accuracy, 'b': b, 'c': c, 'p': p, 'format_pass_rate': curr_best['dev'].valid_parse_rate, 'avg_input_tokens': curr_best['dev'].avg_input_tokens, 'sec_per_question': curr_best['dev'].avg_latency}
            
            log_version(path=log_path, record=log)
            
            best = curr_best['prompt']
            best_id = curr_best['id']
            
            
        else:
            rounds_since_update += 1
            log = {'version_id': curr_best['id'], 'parent_id': best_id, 'round': round + 1, 'status': 'rejected_test', 'reason': 'failed significance test', 'prompt_text': curr_best['prompt'],
                    'diagnosis': curr_best['diagnosis'], 'train_acc': curr_best['train_acc'], 'dev_acc': curr_best['dev'].accuracy, 'b': b, 'c': c, 'p': p,
                    'format_pass_rate': curr_best['dev'].valid_parse_rate, 'avg_input_tokens': curr_best['dev'].avg_input_tokens, 'sec_per_question': curr_best['dev'].avg_latency
                    }
            log_version(path=log_path, record=log)

        if rounds_since_update >= patience:
            return best, "patience"
    
    return best, "rounds"
      
def log_version(path, record: dict):
    
    with open(path, "a", encoding='utf-8') as f:
        
        f.write(json.dumps(record))
        f.write("\n")