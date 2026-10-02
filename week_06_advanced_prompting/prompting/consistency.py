from collections import Counter
from dataclasses import dataclass
import math

@dataclass
class SCResults:
    answer: float | None
    status: str          # ok, tie, insufficient_samples
    vote_share_valid: float | None
    vote_share_all: float   | None
    margin: float   | None
    entropy: float  | None
    is_tie: bool
    n_samples: int
    n_valid: int
    distribution: dict
    samples: list
    latency_s: float
    input_tokens: int
    output_tokens: int
    stop_reasons: list
    llm_resp: list


def self_consistency(llm, system, messages, parser, n = 5, temperature=0.7, max_tokens=1024, base_seed= 0, min_valid = 3, early_stop=False) -> SCResults:
    
    
    results = {}
    samples = []
    majority_votes = (None, 0)
    invalid = 0
    total = 0
    total_latency = 0
    total_input_tokens = 0
    total_output_tokens = 0
    raw_samples = []
    llm_resp = []
    stop_reasons = []
    for i in range(n):
        
        seed = base_seed + i
        
        resp = llm.complete(system=system, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
        stop_reasons.append(resp.stop_reason)
        llm_resp.append(resp)
        raw_samples.append(resp.text)
        total_latency += resp.latency_s
        total_input_tokens += resp.input_tokens
        total_output_tokens += resp.output_tokens
        total += 1
        
        if resp.stop_reason != "end_turn":
            invalid += 1
            continue
        
        parsed = parser(resp.text)
        
        if not parsed.ok:
            invalid += 1
            continue
        
        ans = round(parsed.value, 3)
        samples.append(ans)
        
        if ans not in results:
            results[ans] = 0
        
        results[ans] += 1
        
        if results[ans] > majority_votes[1]:
            
            majority_votes = (ans, results[ans])
        
        
        remaining = n - total
        
        if results[ans] > remaining:
            counts = Counter(samples)
            common = counts.most_common()
            if len(common) == 1:
                next = 0
            else:
                next = common[1][1]
            if common[0][1] - next > remaining and common[0][1] >= min_valid and early_stop:
                break
    
    counts = Counter(samples)
    common = counts.most_common()
    valid = total - invalid 
    vote_share_valid = majority_votes[1] / valid if valid > 0 else None
    vote_share_all = majority_votes[1] / total
    is_tie = False
    
    if len(common) > 1:
        is_tie = common[0][1] == common[1][1]
        
    margin = calculate_margin(common, valid)
    entropy = calculate_entropy(common, valid)

    status = "ok"
    
    if is_tie:
        status = "tie"
    
    if valid < min_valid:
        status = "insufficient_samples"
    
    answer = majority_votes[0]
    if status == "insufficient_samples":
        answer = None
    
    #breakpoint()
    return SCResults(answer=answer, status=status, vote_share_valid=vote_share_valid, vote_share_all=vote_share_all, margin=margin, entropy= entropy, is_tie=is_tie, n_samples= total,
                     n_valid = valid, distribution=to_dict(common), samples=raw_samples, latency_s=total_latency, input_tokens=total_input_tokens, output_tokens=total_output_tokens, llm_resp=llm_resp, stop_reasons=stop_reasons)
    


def to_dict(common):
    
    if len(common) < 1:
        return {}
    
    res = {}
    for k, v in common:
        res[k] = v
    
    return res

def calculate_margin(total_counts, valid):

    if valid == 0:
        return None
    if len(total_counts) == 1:
        second = 0
    else:
        second = total_counts[1][1]
        
    return (total_counts[0][1] - second) / valid

def calculate_entropy(total_counts, valid):
    
    p = [total_counts[i][1] / valid for i in range(len(total_counts))] if valid > 0 else None
    
    if p is None:
        return None
    
    if valid == 0:
        return None
    entropy = -sum([v * math.log2(v) for v in p]) / math.log2(valid) if math.log2(valid) > 0 else 0
        
    return entropy