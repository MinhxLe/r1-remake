from typing import TypedDict

from r1 import os_utils
import re
from r1.data.core import Split, Chat
from datasets import load_dataset
import math
from dataclasses import dataclass


class Task(TypedDict):
    nums: list[int]
    target: int


class ExtraInfo(TypedDict):
    idx: int


class DatasetRow(TypedDict):
    prompt: str | list[Chat]
    task: Task
    extra_info: ExtraInfo


@dataclass
class Response:
    answer: str
    rationale: str | None


def _create_prompt(task: Task) -> str:
    nums, target = task["nums"], task["target"]
    return f"""You area a helpful assistant. You will think about the reasoning process for the problem and then provides the user with the answer.Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags and return the final answer in <answer> </answer> tags - for example <answer> (1 + 2) / 3 </answer>."""


def _create_instruct_prompt(task: Task) -> list[Chat]:
    nums, target = task["nums"], task["target"]
    return [
        dict(
            role="system",
            content="You are a helpful assistant. You first think about the reasoning process and then provide the user with the final answer.",
        ),
        dict(
            role="user",
            content=f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
        ),
    ]


def extract_response(response_str) -> Response | None:
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, response_str, re.DOTALL)
    matches = list(match)
    if len(matches) == 1:
        answer = matches[-1].group(1).strip()
    else:
        answer = None

    rationale_pattern = r"<think>(.*?)</think>"
    match = re.finditer(rationale_pattern, response_str, re.DOTALL)
    matches = list(match)
    if len(matches) == 1:
        rationale = matches[-1].group(1).strip()
    else:
        rationale = None
    if answer is not None:
        return Response(answer=answer, rationale=rationale)
    else:
        return None


def get_dataset(split: Split, use_instruct_prompt: bool):
    # [TODO] add split implementation
    raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    def process_fn(task: Task, idx: int) -> DatasetRow:
        if use_instruct_prompt:
            prompt = _create_instruct_prompt(task)
        else:
            raise NotImplementedError
        return {
            "prompt": prompt,
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
    response = extract_response(response_str)
    if response is None:
        return 0

    equation_str = response.answer
    if equation_str is None:
        return 0

    if not _validate_equation(equation_str, task["nums"]):
        return fmt_score

    answer = _evaluate_equation(equation_str)
    if answer is None:
        # seems a little weird to give formatting score
        # when equation doesn't parse. maybe this should be 0?
        return fmt_score

    if math.isclose(answer, task["target"]):
        return score

    return fmt_score
