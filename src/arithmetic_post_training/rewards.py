from .arithmetic import parse_answer


def correctness_reward(completions, answer, **kwargs):
    """Reward a plain-text response containing the correct integer."""
    return [
        float(parse_answer(text) == int(expected))
        for text, expected in zip(completions, answer)]