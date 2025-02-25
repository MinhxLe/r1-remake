import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from dataclasses import dataclass, field
from typing import List, Iterable
from torch.nn.utils.rnn import pad_sequence
import logging
import json
from datetime import datetime
from r1.data.countdown import get_dataset, compute_score, Task
import copy
import random
from contextlib import nullcontext

@dataclass
class ResponseGroup:
    responses: List[str]
    response_token_ids: List[torch.Tensor]
    log_probs_list: List[torch.Tensor]
    generated_token_id_start: int
    device: torch.device

    sequence_mask: torch.Tensor = field(init=False)
    log_probs_tensor: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Process the sequence mask and log probs after initialization
        self.sequence_mask = pad_sequence(
            [torch.ones_like(lp, device=self.device) for lp in self.log_probs_list],
            batch_first=True,
            padding_value=0
        ).detach()

        self.log_probs_tensor = pad_sequence(
            self.log_probs_list,
            batch_first=True,
            padding_value=0
        ).detach()


@dataclass
class GRPOIterationMetrics:
    objective: float
    policy_objective: float
    kl_div: float
    mean_reward: float
    fraction_correct: float
    mean_reply_length: float
    sample_response: str


@dataclass
class GRPOConfig:

    seed: int = 42

    # Model configs
    model_name: str = "unsloth/Llama-3.2-1B-Instruct"
    model_temperature: float = 0.7

    # Generation configs
    max_new_tokens: int = 500

    # Training configs
    batches_per_iteration: int = 2
    train_batch_size: int = 2
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 1

    # GRPO specific configs
    response_group_size: int = 8  # G in the paper
    epsilon: float = 0.2  # ε for clipping
    beta: float = 0.01   # KL divergence penalty coefficient
    mu: int = 5         # Number of optimization steps per prompt

    # Reward configs
    format_score: float = 0.1
    solve_score: float = 1.0

    # Logging configs
    log_every_n_steps: int = 1
    eval_every_n_steps: int = 50
    save_model_every_n_steps: int = 50
    save_generations_every_n_steps: int = 1
    generation_log_file: str = "generations.jsonl"

class CountdownGRPO:
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_model()

        self.train_dataset = get_dataset("train")
        self.test_dataset = get_dataset("test")

        self.wandb = os.getenv("WANDB_API_KEY")

    def _init_model(self):
        """Initialize the model and tokenizer using transformers"""

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        outer_iteration_count = 0
        best_eval_reward = 0
        random.seed(self.config.seed)

        if self.wandb:
            wandb.init(
                project="countdown-grpo",
                config=vars(config),
                name=f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        for _ in range(self.config.num_epochs):
            batched_training_index_list = self._create_batched_training_index_list(seed=random.randint(0, 1e6))

            for batched_training_indices in batched_training_index_list:

                # Perform GRPO update
                metrics = self._outer_iteration(batched_training_indices, optimizer)
                outer_iteration_count += 1

                if outer_iteration_count % self.config.log_every_n_steps == 0:
                    logger.info(f"Iteration {outer_iteration_count} metrics: {metrics}")
                    if self.wandb:
                        wandb.log({
                            "objective": metrics.objective,
                            "policy_objective": metrics.policy_objective,
                            "kl_div": metrics.kl_div,
                            "mean_reward": metrics.mean_reward,
                            "fraction_correct": metrics.fraction_correct,
                            "mean_reply_length": metrics.mean_reply_length,
                            "iteration": outer_iteration_count
                        })

                if outer_iteration_count % self.config.save_generations_every_n_steps == 0:
                    logger.info(f"Iteration {outer_iteration_count} response: {metrics.sample_response}")
                    with open(self.config.generation_log_file, "a") as f:
                        json.dump({
                            "iteration": outer_iteration_count,
                            "response": metrics.sample_response
                        }, f)
                        f.write("\n")

                if outer_iteration_count % self.save_model_every_n_steps == 0:
                    self.model.save_pretrained(f"model_{outer_iteration_count}")
                    self.tokenizer.save_pretrained(f"model_{outer_iteration_count}")

                #TODO: Evaluate on held out sample
                #and save model if it's good!



    def _outer_iteration(self, batched_training_indices: Iterable[Iterable[int]], optimizer: torch.optim.Optimizer) -> GRPOIterationMetrics:
        """Perform one outer iteration of GRPO"""

        reference_model = copy.deepcopy(self.model)

        for training_indices in batched_training_indices:
            training_batch = self.train_dataset.select(training_indices)

            # implicit iteration over elements of training_batch
            batch_response_groups = []
            batch_rewards = []
            batch_advantages = []
            batch_ref_log_probs = []
            # iteration over all prompts holding pi_theta_old model constant
            # potentially could relax this and let theta_old vary with task
            for task in training_batch:
                response_group = self._generate_response_group(
                    task["prompt"][0]["content"],
                    self.config.response_group_size,
                    self.config.max_new_tokens,
                )

                batch_response_groups.append(response_group)
                rewards = self._compute_rewards(response_group.responses, task)
                batch_rewards.append(rewards)
                batch_advantages.append(self._normalize_advantages(rewards))
                batch_ref_log_probs.append(self._compute_log_probs(reference_model, response_group, no_grad = True).detach())

            torch.cuda.empty_cache()
            self.model.train()
            self.model.gradient_checkpointing_enable()

            for response_group, rewards, advantages, ref_log_probs in zip(batch_response_groups, batch_rewards, batch_advantages, batch_ref_log_probs):
                for _ in range(self.config.mu):

                    new_log_probs = self._compute_log_probs(self.model, response_group)
                    ratio = torch.exp(new_log_probs - response_group.log_probs_tensor)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.config.epsilon,
                        1 + self.config.epsilon
                    )

                    expanded_advantages = advantages.unsqueeze(-1).expand_as(ratio)

                    policy_objective_matrix = torch.min(
                        ratio * expanded_advantages,
                        clipped_ratio * expanded_advantages
                    )

                    policy_objective_by_response = policy_objective_matrix.multiply(response_group.sequence_mask).sum(dim=1).div(response_group.sequence_mask.sum(dim=1))
                    policy_objective = policy_objective_by_response.mean()

                    kl_div = self._compute_kl_div_for_group(ref_log_probs = ref_log_probs, log_probs = new_log_probs, sequence_mask = response_group.sequence_mask)

                    objective = policy_objective - self.config.beta * kl_div
                    loss = -objective

                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.model.gradient_checkpointing_disable()

        flattened_rewards = [reward for rewards in batch_rewards for reward in rewards]
        flattened_reply_lengths = [reply_length for response_group in batch_response_groups for reply_length in response_group.sequence_mask.sum(dim=1).tolist()]

        return GRPOIterationMetrics(objective= objective.item(),
            policy_objective = policy_objective.item(),
            kl_div = kl_div.item(),
            mean_reward = sum(flattened_rewards)/len(flattened_rewards),
            fraction_correct = sum([1 for reward in flattened_rewards if reward == self.config.solve_score]) / len(flattened_rewards),
            mean_reply_length = sum(flattened_reply_lengths) / len(flattened_reply_lengths),
            sample_response = random.choice([response for response_group in batch_response_groups for response in response_group.responses]))

    @torch.no_grad()
    def _generate_response_group(
        self,
        prompt: str,
        num_samples: int,
        max_new_tokens: int = 1000
    ) -> ResponseGroup:
        """Generate multiple responses and their log probs for a single prompt"""

        self.model.eval()

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        responses = []
        response_token_ids = []
        log_probs = []

        #TODO: change to batch eval but watch padding and indexing
        for _ in range(num_samples):
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.model_temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True
            )

            response_ids = outputs.sequences[0]
            responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
            logits = torch.stack(outputs.logits)  # [new_tokens, 1, vocab_size]
            token_log_probs = torch.log_softmax(logits.squeeze(1), dim=-1)  # [new_tokens, vocab_size]
            step_log_probs = torch.gather(
                token_log_probs,
                dim=-1,
                index=response_ids[input_ids.shape[1]:].unsqueeze(-1)
            ).squeeze(-1)
            log_probs.append(step_log_probs)
            response_token_ids.append(response_ids)

        return ResponseGroup(responses=responses, response_token_ids=response_token_ids, log_probs_list=log_probs, generated_token_id_start=input_ids.shape[1], device=self.device)


    def _compute_rewards(self, responses: List[str], task: Task) -> torch.Tensor:
        """Compute rewards for a group of responses"""

        return torch.tensor([compute_score(
                response,
                task,
                self.config.format_score,
                self.config.solve_score
            ) for response in responses], device=self.device)

    def _normalize_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards within a group to get advantages"""
        mean = rewards.mean()
        std = rewards.std()
        if std == 0:
            return torch.zeros_like(rewards)
        return (rewards - mean) / std

    def _compute_kl_div_for_group(self, ref_log_probs: torch.Tensor, log_probs: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence using the unbiased estimator from the paper"""

        ratio = torch.exp(ref_log_probs - log_probs)
        unmasked_expression = ratio - torch.log(ratio) - 1
        masked_expression = unmasked_expression.multiply(sequence_mask).sum(dim=1).div(sequence_mask.sum(dim=1))
        return masked_expression.mean()

    def _compute_log_probs(self, model: AutoModelForCausalLM, response_group: ResponseGroup, no_grad: bool = False) -> torch.Tensor:
        """Compute log probabilities for a list of responses using the current model."""

        if no_grad:
            model.eval()
            context = torch.no_grad()
        else:
            context = nullcontext()

        with context:
            # Need to iterate here for OOM safety
            log_probs_list = []
            for response_ids in response_group.response_token_ids:
                input_ids = response_ids.unsqueeze(0)
                outputs = model(input_ids, labels=input_ids)
                token_log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
                log_probs_list.append(torch.gather(
                    token_log_probs,
                    dim=-1,
                    index=input_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1).squeeze(0)[(response_group.generated_token_id_start-1):])

        return pad_sequence(log_probs_list, batch_first=True, padding_value=0)

    def _create_batched_training_index_list(self, seed: int) -> List[List[int]]:
        """
        Training method that creates batches of indices from the training dataset.
        The method creates batches of size self.config.train_batch_size
        and groups them into iterations of size self.config.batches_per_iteration.
        """

        all_indices = list(range(len(self.train_dataset)))

        # Shuffle indices to randomize training
        random.seed(seed)
        random.shuffle(all_indices)

        # Create batches of size train_batch_size
        batches = []
        for i in range(0,len(self.train_dataset),self.config.train_batch_size):
            batches.append(all_indices[i:i + self.config.train_batch_size])

        batched_training_index_list = []
        # Group batches into iterations of size batches_per_iteration
        for i in range(0, len(batches), self.config.batches_per_iteration):
            batched_training_index_list.append(batches[i:i + self.config.batches_per_iteration])

        return batched_training_index_list

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = GRPOConfig()
    trainer = CountdownGRPO(config)
    trainer.train()
