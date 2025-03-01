import functools
from datasets import Dataset
import torch
from typing import Callable, Any
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from r1.data.core import Chat
import wandb
from r1.utils import model_utils
from r1.data.countdown import get_dataset, compute_score
import random
from loguru import logger
from tqdm import tqdm
from datetime import datetime


@dataclass
class Cfg:
    seed: int = 42

    # Model configs
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-instruct"
    # model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    # sampling
    # model_temperature: float = 0.7

    # Generation configs
    max_new_tokens: int = 500

    # Training configs
    train_batch_size: int = 2
    test_batch_size: int = 8
    lr: float = 1e-5
    n_epochs: int = 1

    # GRPO specific configs
    group_size: int = 2  # G in the paper
    epsilon: float = 0.2  # ε for clipping
    beta: float = 0.1  # KL divergence penalty coefficient
    mu: int = 5  # Number of optimization steps per prompt
    ref_model_update_interval: int = 10000  # how frequently do we update ref model (M)

    # Reward configs
    format_score: float = 0.1
    solve_score: float = 1.0

    # Logging configs
    train_log_interval: int = 1
    eval_interval: int = 100
    save_interval: int = 100
    log_wandb: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Cfg()


@dataclass
class Group:
    prompt_token_ids: torch.Tensor
    response_token_ids: torch.Tensor
    response_masks: torch.Tensor
    response_log_probs: torch.Tensor

    def prompt_length(self):
        return self.prompt_token_ids.shape[1]

    def size(self) -> int:
        return self.response_token_ids.shape[0]

    def max_response_length(self):
        return self.response_token_ids.shape[-1]


@dataclass
class TrainMetric:
    mean_response_length: float
    mean_reward: float
    mean_kl: float
    mean_loss: float


@dataclass
class GRPOTrainer:
    cfg: Cfg
    train_dataset: Dataset
    test_dataset: Dataset
    model: AutoModelForCausalLM = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)

    def __post_init__(self):
        # everything to set up experiment is done as part of initialization so we only have
        # one place where config maps to setup.
        cfg = self.cfg

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            padding_side="left",  # to pad uneven input
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self):
        cfg = self.cfg
        model = self.model
        train_dataset = self.train_dataset
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        # TODO, maybe we should pre tokenize prompts
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )
        ref_model = model_utils.copy_model(model).to(cfg.device)
        ref_model.eval()

        target_model = model_utils.copy_model(model).to(cfg.device)
        target_model.eval()

        if cfg.log_wandb:
            wandb.init(
                project="countdown-grpo",
                config=vars(cfg),
                name=f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

        model.gradient_checkpointing_enable()
        random.seed(cfg.seed)
        torch.random.manual_seed(cfg.seed)

        for epoch in range(cfg.n_epochs):
            for i, batch_tasks in tqdm(enumerate(train_dataloader)):
                step = epoch * cfg.n_epochs + i
                if (step % cfg.ref_model_update_interval) == 0:
                    logger.debug("updating ref model")
                    model_utils.sync_model(model, ref_model)
                model_utils.sync_model(model, target_model)
                # TODO operate over full batch of tasks
                total_n_correct = 0
                total_loss = 0
                total_response_length = 0
                for task in batch_tasks:
                    prompt = task["prompt"]
                    group = self.sample_group(target_model, prompt)
                    group_responses = self.tokenizer.batch_decode(
                        group.response_token_ids, skip_special_tokens=True
                    )
                    logger.debug("sampling group")
                    logger.debug(f"TASK: {task}")
                    logger.debug("RESPONSES: ")
                    for response_i, response in enumerate(group_responses):
                        logger.debug(f"RESPONSES: {response_i}")
                        logger.debug(response)
                    rewards = torch.tensor(
                        [
                            compute_score(
                                response, task, cfg.format_score, cfg.solve_score
                            )
                            for response in group_responses
                        ],
                        dtype=torch.bfloat16,
                    )
                    # the entire trajectory has the same advantage
                    normalized_advantages = (
                        ((rewards - rewards.mean()) / (rewards.std() + 1e-5))
                        .unsqueeze(-1)  # adding seq_len dimension
                        .repeat((1, group.max_response_length()))
                    ).to(cfg.device)

                    with torch.no_grad():
                        ref_log_probs = self.compute_response_log_probs(
                            ref_model, group
                        )

                    for _ in range(cfg.mu):
                        total_loss += self._update_model(
                            optimizer,
                            model,
                            ref_log_probs,
                            group,
                            normalized_advantages,
                        )
                        torch.cuda.empty_cache()  # Explicitly reclaim freed memory
                    __import__("ipdb").set_trace()

                    total_n_correct += (rewards == cfg.solve_score).sum().item()
                    total_response_length += group.response_masks.sum().item()
                metrics = dict(
                    step=step,
                    epoch=epoch,
                    batch=i,
                    train_accuracy=total_n_correct
                    / (cfg.train_batch_size * cfg.group_size),
                    train_mean_response_length=(
                        total_response_length / (cfg.train_batch_size * cfg.group_size)
                    ),
                    mean_train_loss=(total_loss / (cfg.train_batch_size * cfg.mu)),
                )
                if cfg.log_wandb:
                    wandb.log(metrics)
                if (step % cfg.train_log_interval) == 0:
                    logger.info(metrics)
                # TODO add saving model
                # TODO add eval

            if cfg.log_wandb:
                wandb.finish()

    def _update_model(
        self, optimizer, model, ref_log_probs, group, normalized_advantages
    ):
        model_log_probs = self.compute_response_log_probs(model, group)
        ratio = torch.exp(model_log_probs - group.response_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - cfg.epsilon, 1 + cfg.epsilon)
        policy_objectives = torch.min(
            ratio * normalized_advantages,
            clipped_ratio * normalized_advantages,
        )  # (group_size, max_response_length)
        # kl estimator
        kls = (
            torch.exp(ref_log_probs - model_log_probs)
            - (ref_log_probs - model_log_probs)
            - 1
        )
        logger.debug(f"KL VALUE {kls.mean().item()}")
        all_objectives = (
            policy_objectives - cfg.beta * kls
        )  # (group_size, max_response_length)
        objective = (
            # mean per response
            (all_objectives * group.response_masks).sum(dim=1)
            / group.response_masks.sum(dim=1)
        ).mean()  # mean over group
        loss = -objective
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def compute_response_log_probs(self, model, group: Group) -> torch.Tensor:
        token_ids = torch.concat(
            (
                group.prompt_token_ids.expand(group.size(), -1).to(cfg.device),
                group.response_token_ids.to(cfg.device),
            ),
            dim=1,
        )
        output = model(token_ids)
        logits = output.logits[:, group.prompt_length() :]
        log_probs = torch.log_softmax(logits, dim=1)
        # selecting indices of log_probs on the last column
        response_log_probs = torch.gather(
            log_probs, dim=-1, index=group.response_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        assert response_log_probs.shape == group.response_log_probs.shape
        return response_log_probs

    @torch.no_grad()
    def sample_group(self, model, prompt: list[Chat]) -> Group:
        tokenizer, cfg = self.tokenizer, self.cfg
        group_size, max_new_tokens = cfg.group_size, cfg.max_new_tokens

        model.eval()
        prompt_token_ids = tokenizer([prompt], return_tensors="pt").input_ids
        generation = model.generate(
            prompt_token_ids.expand((group_size, -1)).to(
                cfg.device
            ),  # repeat group size times
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )
        prompt_length = prompt_token_ids.shape[-1]
        response_token_ids = generation.sequences[:, prompt_length:]
        response_masks = response_token_ids != tokenizer.pad_token_id
        response_logits = torch.stack(generation.logits).swapaxes(
            0, 1
        )  # swapping sequence and group indices
        log_probs = torch.log_softmax(response_logits, -1)
        response_log_probs = torch.gather(
            log_probs, dim=-1, index=response_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        return Group(
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            response_log_probs=response_log_probs,
            response_masks=response_masks,
        )


if __name__ == "__main__":
    cfg = Cfg()
    trainer = GRPOTrainer(
        cfg=cfg,
        test_dataset=get_dataset("test"),
        train_dataset=get_dataset("train"),
    )
    trainer.train()
