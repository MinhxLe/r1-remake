from typing import TypeVar
import copy
from torch import nn


ModelT = TypeVar("ModelT", bound=nn.Module)


def copy_model(model: ModelT) -> ModelT:
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(model.state_dict())
    return new_model


def sync_model(source_model, target_model):
    source_model_dict = source_model.state_dict()
    target_model_dict = target_model.state_dict()
    for key in source_model_dict:
        target_model_dict[key] = source_model_dict[key]
    target_model.load_state_dict(target_model_dict)
    return target_model
