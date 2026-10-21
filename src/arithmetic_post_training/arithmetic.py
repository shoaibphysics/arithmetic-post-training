import random
import re


def make_arithmetic_data(n_train=5000, n_val=500, n_test=1000, seed=42):
    """Shuffle unique arithmetic questions into train, validation, and test sets."""
    examples = []

    for a in range(10, 100):
        for b in range(10, 100):
            for op in ("+", "-"):
                answer = a + b if op == "+" else a - b
                examples.append({
                    "prompt": f"Question: What is {a} {op} {b}?\nAnswer:",
                    "completion": f" {answer}"})

    total = n_train + n_val + n_test
    if min(n_train, n_val, n_test) < 1 or total > len(examples):
        raise ValueError("Choose positive split sizes totaling at most 16,200.")

    random.Random(seed).shuffle(examples)

    return {
        "train": examples[:n_train],
        "validation": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:total]}


def parse_answer(text):
    """Return an integer for a valid response, otherwise None."""
    text = text.strip()
    return int(text) if re.fullmatch(r"[+-]?[0-9]+", text) else None


def make_harder_data(n_train=512, n_val=64, n_test=200, seed=42):
    """Create unique ordered questions for four harder arithmetic tasks."""
    total = n_train + n_val + n_test
    if min(n_train, n_val, n_test) < 1 or total > 8100:
        raise ValueError("Choose positive split sizes totaling at most 8,100.")

    rng = random.Random(seed)
    ranges = {
        "add_sub_3digit": (100, 999),
        "add_sub_4digit": (1000, 9999),
        "multiplication": (10, 99),
        "two_step": (10, 99),
    }
    tasks = {}

    for task, (low, high) in ranges.items():
        examples = {}

        while len(examples) < total:
            a, b = rng.randint(low, high), rng.randint(low, high)

            if task == "multiplication":
                expression, answer = f"{a} * {b}", a * b

            elif task == "two_step":
                c = rng.randint(2, 9)
                expression, answer = f"({a} + {b}) * {c}", (a + b) * c

            else:
                op = rng.choice(("+", "-"))
                expression = f"{a} {op} {b}"
                answer = a + b if op == "+" else a - b

            examples[expression] = {
                "prompt": f"Question: What is {expression}?\nAnswer:",
                "completion": f" {answer}",
            }

        rows = list(examples.values())
        tasks[task] = {
            "train": rows[:n_train],
            "validation": rows[n_train:n_train + n_val],
            "test": rows[n_train + n_val:total],
        }

    return tasks