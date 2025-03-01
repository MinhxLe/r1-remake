from typing import Literal, TypedDict

Role = Literal["user", "assistant"]
Split = Literal["train", "test"]


class Chat(TypedDict):
    role: Role
    content: str
