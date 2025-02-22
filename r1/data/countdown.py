from typing import TypedDict

from numpy import format_float_scientific
from r1 import os_utils
import re
from r1.data.core import Split, Chat, extract_task_response
from datasets import load_dataset
import math


class Task(TypedDict):
    nums: list[int]
    target: int


class ExtraInfo(TypedDict):
    idx: int


class DatasetRow(TypedDict):
    prompt: list[Chat]
    task: Task
    extra_info: ExtraInfo


def _create_prefix(task: Task) -> str:
    nums, target = task["nums"], task["target"]
    return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""


def get_dataset(split: Split):
    # [TODO] add split implementation
    raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)

    def process_fn(task: Task, idx: int) -> DatasetRow:
        return {
            "prompt": [{"role": "user", "content": _create_prefix(task)}],
            "task": task,
            "extra_info": {"idx": idx},
        }

    dataset = raw_dataset.map(
        process_fn,
        with_indices=True,
        num_proc=os_utils.n_cores(),
    )
    return dataset


def _validate_equation(equation_str: str, valid_nums: list[int]) -> bool:
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        has_only_valid_symbols = re.match(allowed_pattern, equation_str) is not None

        # Each number should be used exactly once
        nums_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        nums_used_exactly_once = sorted(nums_in_eq) == sorted(valid_nums)

        return has_only_valid_symbols and nums_used_exactly_once

    except Exception:
        return False


def _evaluate_equation(equation_str: str) -> int | None:
    try:
        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def compute_score(
    response_str: str, task: Task, fmt_score: float, score: float
) -> float:
    response = extract_task_response(response_str)
    if response is not None:
        equation_str = response.answer
        if equation_str is not None and _validate_equation(equation_str, task["nums"]):
            answer = _evaluate_equation(equation_str)
            if answer is not None:
                if math.isclose(answer, task["target"]):
                    computed_score = score
                else:
                    computed_score = fmt_score
            else:
                computed_score = fmt_score
        else:
            computed_score = fmt_score
    else:
        computed_score = 0

    return computed_score
