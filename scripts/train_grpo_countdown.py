from datasets import Dataset
import torch
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
# from vllm import LLM, SamplingParams


@dataclass
class Cfg:
    seed: int = 42

    # Model configs
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_instruct_prompt: bool = True

    # sampling
    # model_temperature: float = 0.7

    # Generation configs
    max_new_tokens: int = 500

    # Training configs
    train_batch_size: int = 8
    test_batch_size: int = 8
    lr: float = 1e-5
    n_epochs: int = 1

    # GRPO specific configs
    group_size: int = 5  # G in the paper
    epsilon: float = 0.2  # ε for clipping
    beta: float = 0.001  # KL divergence penalty coefficient
    mu: int = 1  # Number of optimization steps per prompt
    ref_model_update_interval: int = 20  # how frequently do we update ref model (M)

    # Reward configs
    format_score: float = 0.1
    solve_score: float = 1.0

    # Logging configs
    train_log_interval: int = 1
    eval_interval: int = 100
    save_interval: int = 100
    log_wandb: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Cfg()


@dataclass
class Group:
    prompt_token_ids: torch.Tensor
    response_token_ids: torch.Tensor
    response_masks: torch.Tensor
    response_log_probs: torch.Tensor | None

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

        random.seed(cfg.seed)
        torch.random.manual_seed(cfg.seed)

        for epoch in range(cfg.n_epochs):
            for i, batch_rows in tqdm(enumerate(train_dataloader)):
                step = epoch * cfg.n_epochs + i
                if (step % cfg.ref_model_update_interval) == 0:
                    model_utils.sync_model(model, ref_model)
                model_utils.sync_model(model, target_model)
                # TODO operate over full batch of tasks
                total_n_correct = 0
                total_loss = 0
                total_response_length = 0
                total_reward = 0
                for row in batch_rows:
                    prompt = row["prompt"]
                    task = row["task"]
                    group = self.sample_group(target_model, prompt)
                    group_responses = self.tokenizer.batch_decode(
                        group.response_token_ids, skip_special_tokens=True
                    )
                    rewards = torch.tensor(
                        [
                            compute_score(
                                response, task, cfg.format_score, cfg.solve_score
                            )
                            for response in group_responses
                        ],
                        dtype=torch.bfloat16,
                    )
                    logger.debug("sampling group")
                    logger.debug(f"TASK: {task}")
                    logger.debug("RESPONSES: ")
                    for response_i, (response, reward) in enumerate(
                        zip(group_responses, rewards)
                    ):
                        logger.debug(f"response: {response_i}, reward: {reward}")
                        logger.debug(response)
                    normalized_advantages = (
                        ((rewards - rewards.mean()) / (rewards.std() + 1e-5))
                        .unsqueeze(-1)  # adding seq_len dimension
                        .repeat((1, group.max_response_length()))
                    ).to(cfg.device)

                    with torch.no_grad():
                        ref_log_probs = self.compute_response_log_probs(
                            ref_model, group
                        )

                    model.train()
                    model.gradient_checkpointing_enable()
                    for _ in range(cfg.mu):
                        total_loss += self._update_model(
                            optimizer,
                            model,
                            ref_log_probs,
                            group,
                            normalized_advantages,
                        )
                        torch.cuda.empty_cache()  # Explicitly reclaim freed memory
                    model.gradient_checkpointing_disable()
                    total_n_correct += (rewards == cfg.solve_score).sum().item()
                    total_response_length += group.response_masks.sum().item()
                    total_reward += rewards.sum().item()
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
                    mean_reward=(
                        total_reward / (cfg.train_batch_size * cfg.group_size)
                    ),
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
        all_objectives = (
            policy_objectives - cfg.beta * kls
        )  # (group_size, max_response_length)
        objective = (
            # mean per response
            (all_objectives * group.response_masks).sum(dim=1)
            / group.response_masks.sum(dim=1)
        ).mean()  # mean over group

        mean_policy_objective = (
            policy_objectives.sum() / group.response_masks.sum()
        ).item()
        mean_kl = (kls.sum() / group.response_masks.sum()).item()
        logger.debug(
            dict(
                mean_policy_objective=mean_policy_objective,
                mean_kl=mean_kl,
            )
        )
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
    def sample_group_v2(self, model, prompt: list[Chat]) -> Group:
        tokenizer, cfg = self.tokenizer, self.cfg
        group_size, max_new_tokens = cfg.group_size, cfg.max_new_tokens

        # Initialize vLLM if not already done
        if not hasattr(self, "vllm_engine"):
            self.vllm_engine = LLM(
                model=model.cfg.model_name,
                dtype="auto",
                gpu_memory_utilization=0.8,
                tensor_parallel_size=1,  # adjust based on your GPU setup
            )
            self.sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                n=group_size,  # number of generations per prompt
            )

        # Format prompt for vLLM
        if cfg.use_instruct_prompt:
            formatted_prompt = tokenizer.apply_chat_template(
                [prompt], return_tensors="pt"
            )
            # Convert to string for vLLM
            formatted_prompt = tokenizer.decode(
                formatted_prompt[0], skip_special_tokens=False
            )
        else:
            formatted_prompt = prompt

        # Generate responses using vLLM
        outputs = self.vllm_engine.generate([formatted_prompt], self.sampling_params)

        # Process the outputs
        prompt_token_ids = tokenizer([formatted_prompt], return_tensors="pt").input_ids
        prompt_length = prompt_token_ids.shape[-1]

        # Extract generated sequences, convert to token IDs
        generations = [output.outputs[0].text for output in outputs]
        all_token_ids = []
        all_logprobs = []
        max_length = 0

        # Process each generated output
        for gen in generations:
            gen_tokens = tokenizer(gen, return_tensors="pt").input_ids[0]
            all_token_ids.append(gen_tokens)
            max_length = max(max_length, len(gen_tokens))

        # Pad sequences to the same length
        response_token_ids = torch.full(
            (group_size, max_length), tokenizer.pad_token_id
        )
        response_masks = torch.zeros((group_size, max_length), dtype=torch.bool)

        for i, tokens in enumerate(all_token_ids):
            response_token_ids[i, : len(tokens)] = tokens
            response_masks[i, : len(tokens)] = True
        return Group(
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            response_log_probs=None,
            response_masks=response_masks,
        )

    @torch.no_grad()
    def sample_group(self, model, prompt: list[Chat]) -> Group:
        tokenizer, cfg = self.tokenizer, self.cfg
        group_size, max_new_tokens = cfg.group_size, cfg.max_new_tokens

        model.eval()
        if cfg.use_instruct_prompt:
            prompt_token_ids = tokenizer.apply_chat_template(
                [prompt], return_tensors="pt"
            )
        else:
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
        train_dataset=get_dataset("train", cfg.use_instruct_prompt),
        test_dataset=get_dataset("test", cfg.use_instruct_prompt),
    )
    # dataset = trainer.train_dataset
    # rewards = []
    # for i, row in tqdm(enumerate(dataset)):
    #     group = trainer.sample_group(trainer.model, row["prompt"])
    #
    #     group_responses = trainer.tokenizer.batch_decode(
    #         group.response_token_ids, skip_special_tokens=True
    #     )
    #     rewards.append(
    #         [
    #             compute_score(response, row["task"], cfg.format_score, cfg.solve_score)
    #             for response in group_responses
    #         ]
    #     )
    #     print(f"task {i / len(dataset)}, reward {rewards[-1]}")
