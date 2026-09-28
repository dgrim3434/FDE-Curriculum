PRICES_PER_MILLION = {
    # Model Name: (input price, output price)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

def call_price(model: str, input_tokens: int, output_tokens: int):
    
    if model not in PRICES_PER_MILLION:
        raise ValueError(f"ERROR: Price not tracked for provided model name: {model}")
    
    in_price, out_price = PRICES_PER_MILLION[model]
    
    return (input_tokens * in_price) / 1_000_000 + (out_price * output_tokens) / 1_000_000