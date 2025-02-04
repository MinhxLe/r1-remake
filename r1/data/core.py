from dataclasses import dataclass
from typing import Literal, TypedDict
import re

Role = Literal["user", "assistant"]
Split = Literal["train", "test"]


class Chat(TypedDict):
    role: Role
    content: str


@dataclass
class TaskResponse:
    answer: str
    rationale: str | None


def format_task(task_str: str) -> str:
    return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {task_str}
Assistant: Let me solve this step by step.
<think>"""


def extract_task_response(response_str) -> TaskResponse | None:
    if "Assistant:" in response_str:
        response_str = response_str.split("Assistant:", 1)[1]
    else:
        return None
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, response_str)
    matches = list(match)
    if len(matches) == 1:
        answer = matches[-1].group(1).strip()
    else:
        answer = None

    rationale_pattern = r"<think>(.*?)</think>"
    match = re.finditer(rationale_pattern, response_str)
    matches = list(match)
    if len(matches) == 1:
        rationale = matches[-1].group(1).strip()
    else:
        rationale = None
    if answer is not None:
        return TaskResponse(answer=answer, rationale=rationale)
    else:
        return None
