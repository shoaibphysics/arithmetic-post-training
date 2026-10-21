from pathlib import Path

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    qwen2_5_chat_template,
)

import json

from datasets import Dataset
from trl import GRPOConfig

from .target_game import game_prompt


def load_agent(adapter, train_new_tokens=False):
    """Load an adapter, optionally enabling four chat/tool token embeddings."""
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.chat_template = qwen2_5_chat_template
    add_response_schema(tokenizer)
    tokenizer.chat_template = get_training_chat_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        dtype=torch.float32,
    ).to(device)

    if train_new_tokens:
        config = LoraConfig.from_pretrained(adapter)
        config.trainable_token_indices = tokenizer.convert_tokens_to_ids([
            "<|im_start|>",
            "<|im_end|>",
            "<tool_call>",
            "</tool_call>",
        ])
        config.inference_mode = False
        model = get_peft_model(model, config)

        # Preserve the old LoRA weights and initialize the extra token rows.
        weights = get_peft_model_state_dict(model)
        weights.update(
            load_file(str(Path(adapter) / "adapter_model.safetensors"))
        )
        set_peft_model_state_dict(model, weights)
    else:
        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=True
        )

    # Compatibility between TRL's output head and PEFT's token wrapper.
    head = model.get_output_embeddings()
    if not hasattr(head, "bias"):
        head.register_parameter("bias", None)

    model.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]

    return model, tokenizer

def game_dataset(rows):
    return Dataset.from_list([
        {
            "prompt": game_prompt(row, include_state=False),
            "numbers": row["numbers"],
            "target": row["target"],
            "task_id": row["task_id"],
        }
        for row in rows
    ])


class EpisodeRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.write_text("")
        self.records = []
        self.__name__ = "game_success"

    def __call__(
        self, completions, completion_ids, environments, task_id,
        trainer_state=None, **kwargs
    ):
        batch = []

        for messages, ids, env, key in zip(
            completions, completion_ids, environments, task_id
        ):
            observed_success = False

            for message in messages:
                if message.get("role") != "tool":
                    continue
                try:
                    observation = json.loads(message.get("content") or "null")
                except (ValueError, TypeError):
                    continue

                observed_success |= (
                    isinstance(observation, dict)
                    and observation.get("done") is True
                    and observation.get("success") is True
                    and observation.get("numbers") == [env.target]
                    and observation.get("target") == env.target
                )

            batch.append({
                "task_id": key,
                "step": getattr(trainer_state, "global_step", 0),
                "reward": float(env.success and observed_success),
                "used_tool": bool(env.history),
                "invalid_actions": sum(not h["valid"] for h in env.history),
                "attempts": env.attempts,
                "completion_tokens": len(ids),
                "messages": messages,
                "history": list(env.history),
            })

        self.records.extend(batch)
        with self.path.open("a") as handle:
            for record in batch:
                handle.write(json.dumps(record) + "\n")

        return [record["reward"] for record in batch]


def game_config(output_dir, **overrides):
    settings = dict(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=256,
        max_tool_calling_iterations=3,
        temperature=1.0, top_p=1.0, top_k=0,
        num_iterations=1, beta=0.0,
        loss_type="grpo", scale_rewards="group",
        disable_dropout=True, gradient_checkpointing=False,
        bf16=False, fp16=False, dataloader_pin_memory=False,
        save_strategy="no", logging_steps=5,
        report_to="none", disable_tqdm=True, seed=42,
    )
    settings.update(overrides)
    return GRPOConfig(output_dir=str(output_dir), **settings)