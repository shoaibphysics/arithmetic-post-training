import pandas as pd
import torch

from .arithmetic import parse_answer


@torch.inference_mode()
def evaluate_arithmetic(
    model,
    tokenizer,
    examples,
    batch_size=16,
    max_new_tokens=8,
    num_generations=1,
    temperature=1.0):
    """Generate answers and measure correctness and valid integer format."""
    was_training = model.training
    padding_side = tokenizer.padding_side
    records = []

    sampling = {}
    if num_generations > 1:
        sampling = dict(temperature=temperature, top_p=1.0, top_k=0)

    try:
        model.eval()
        tokenizer.padding_side = "left"

        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            inputs = tokenizer(
                [row["prompt"] for row in batch],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=num_generations > 1,
                num_return_sequences=num_generations,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **sampling)

            # Decode only newly generated tokens, excluding the padded prompts.
            responses = tokenizer.batch_decode(
                output[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)

            repeated_rows = [row for row in batch for _ in range(num_generations)]

            for row, response in zip(repeated_rows, responses):
                prediction = parse_answer(response)
                expected = int(row["completion"])

                records.append({
                    "prompt": row["prompt"],
                    "response": response,
                    "expected": expected,
                    "correct": prediction == expected,
                    "valid": prediction is not None,
                })
    finally:
        tokenizer.padding_side = padding_side
        model.train(was_training)

    return pd.DataFrame(records)