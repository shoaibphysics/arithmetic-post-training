import json
from copy import deepcopy


import random
from itertools import combinations_with_replacement


class TargetGameEnv:
    def __init__(self, max_attempts=3):
        if type(max_attempts) is not int or max_attempts < 2:
            raise ValueError("max_attempts must be an integer of at least two.")

        self.max_attempts = max_attempts
        self.reset()

    def reset(self, numbers=None, target=None, **kwargs):
        if numbers is not None:
            if len(numbers) != 3 or any(type(n) is not int for n in numbers):
                raise ValueError("A puzzle needs exactly three integers.")
            if type(target) is not int:
                raise ValueError("The target must be an integer.")
        elif target is not None:
            raise ValueError("Provide numbers together with a target.")

        self.numbers = list(numbers) if numbers is not None else []
        self.target = target
        self.attempts = 0
        self.done = False
        self.success = False
        self.history = []

        return self._observation()

    def combine(self, i: int, j: int, op: str) -> str:
        """Combine two available numbers using their zero-based positions.

        Args:
            i: Position of the left operand in the current numbers list.
            j: Position of a different right operand in the current numbers list.
            op: The operation: +, -, or *.

        Returns:
            JSON containing the updated numbers, target, and episode status.
        """
        action = {"i": i, "j": j, "op": op}

        if self.done:
            return self._record(action, False, "Episode finished.")

        self.attempts += 1

        valid_indices = (
            type(i) is int and type(j) is int
            and 0 <= i < len(self.numbers)
            and 0 <= j < len(self.numbers)
            and i != j
        )

        error = None

        if not valid_indices:
            error = "Choose two different valid indices."
        elif op not in ("+", "-", "*"):
            error = "Use +, -, or *."
        else:
            a, b = self.numbers[i], self.numbers[j]
            value = a + b if op == "+" else a - b if op == "-" else a * b

            self.numbers = [
                n for k, n in enumerate(self.numbers) if k not in (i, j)
            ]
            self.numbers.append(value)

        self.success = len(self.numbers) == 1 and self.numbers[0] == self.target
        self.done = len(self.numbers) == 1 or self.attempts >= self.max_attempts

        return self._record(action, error is None, error)

    def _observation(self, error=None):
        state = {
            "numbers": self.numbers,
            "target": self.target,
            "done": self.done,
            "success": self.success,
            "attempts_left": max(0, self.max_attempts - self.attempts),
        }

        if error is not None:
            state["error"] = error

        return json.dumps(state, separators=(",", ":"))

    def _record(self, action, valid, error=None):
        observation = self._observation(error)

        self.history.append({
            "action": deepcopy(action),
            "valid": valid,
            "observation": observation,
        })

        return observation


def _solutions(numbers):
    if len(numbers) == 1:
        yield numbers[0], []
        return

    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i == j:
                continue

            a, b = numbers[i], numbers[j]
            rest = [n for k, n in enumerate(numbers) if k not in (i, j)]

            for op in ("+", "-", "*"):
                value = a + b if op == "+" else a - b if op == "-" else a * b

                for target, tail in _solutions(rest + [value]):
                    yield target, [{"i": i, "j": j, "op": op}] + tail


def make_game_data(n_train=512, n_validation=32, n_test=64, seed=42):
    """Create solvable puzzles, split by their starting numbers."""
    sizes = {"train": n_train, "validation": n_validation, "test": n_test}
    rng = random.Random(seed)

    groups = list(combinations_with_replacement(range(1, 10), 3))
    rng.shuffle(groups)
    partitions = {
        "train": groups[:115],
        "validation": groups[115:140],
        "test": groups[140:],
    }

    result = {}

    for split, triples in partitions.items():
        rows = []

        for numbers in triples:
            solutions = {}

            for target, actions in _solutions(list(numbers)):
                if 1 <= target <= 100:
                    solutions.setdefault(target, []).append(actions)

            for target, paths in sorted(solutions.items()):
                rows.append({
                    "numbers": list(numbers),
                    "target": target,
                    "solution": rng.choice(paths),
                    "task_id": "-".join(map(str, numbers)) + f":{target}",
                })

        if not 1 <= sizes[split] <= len(rows):
            raise ValueError(
                f"{split} size must be between 1 and {len(rows)}."
            )

        rng.shuffle(rows)
        result[split] = rows[:sizes[split]]

    return result


def game_prompt(row, include_state=True):
    """Create task messages; GRPO can append the initial state itself."""
    env = TargetGameEnv()
    state = env.reset(numbers=row["numbers"], target=row["target"])

    return [
        {
            "role": "system",
            "content": (
                "Reach the target by combining two available numbers at a time. "
                "Call combine with two different zero-based indices and +, -, or *. "
                "Each starting number must be used exactly once. "
                "Use the updated numbers after each call. When done is true, reply done."
            ),
        },
        {"role": "user", "content": state if include_state else ""},
    ]


def make_demonstration(row):
    """Build a successful tool conversation for supervised warm-up."""
    from transformers.utils import get_json_schema

    env = TargetGameEnv()
    env.reset(numbers=row["numbers"], target=row["target"])
    messages = game_prompt(row)

    for action in row["solution"]:
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "combine",
                    "arguments": dict(action),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "content": env.combine(**action),
        })

    if not env.success:
        raise ValueError("The supplied demonstration does not solve its puzzle.")

    messages.append({"role": "assistant", "content": "done"})

    return {
        "messages": messages,
        "tools": json.dumps([get_json_schema(env.combine)]),
    }