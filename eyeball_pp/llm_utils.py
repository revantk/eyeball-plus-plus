LLM_PRICING = {
    "gpt-3.5-turbo": (0.002 / 1000.0, 0.002 / 1000.0),
    "gpt-4": (0.03 / 1000.0, 0.06 / 1000.0),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM request"""
    input_token_price, output_token_price = LLM_PRICING.get(model, (0, 0))
    total_cost = (input_token_price * input_tokens) + \
        (output_token_price * output_tokens)
    return total_cost
