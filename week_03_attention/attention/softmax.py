"""
The stable softmax implementation:
    - Guards against overflow errors: happens around 88 for int32. The fix is subtracting by the max of each row before softmax. Addition and subtraction do not affect the exp
    function since the only thing that is measured is the distance between values
    
    - Guards against rows full of -inf. Happens when left padding is used which resuts in NaN. This will propogate throughout the result of the pipeline and result in a crash.
    Any calculation done against NaN results in NaN. The fix is to set rows full of NaN to all zeros.
"""
import torch

def stable_softmax(scores, dims=-1):
    
    # Extracting the row max to ensure no overflow
    row_max = torch.amax(scores, dim=dims, keepdim=True)
    # Converting -inf into 0 to guard against -inf - -inf -> NaN
    row_max = torch.where(torch.isneginf(row_max), torch.zeros_like(row_max), row_max)
    
    # Calculating the exponential function
    exp = torch.exp(scores - row_max)
    
    # Calculating denominatior summing across all columns
    denom = exp.sum(dim=dims, keepdim=True)
    
    return exp / denom.clamp_min_(torch.finfo(exp.dtype).tiny)