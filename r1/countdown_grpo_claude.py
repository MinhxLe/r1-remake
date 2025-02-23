import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import Dataset
import numpy as np
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
import logging
import json
from datetime import datetime
from r1.data.countdown import get_dataset, compute_score, Task

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
class GRPOConfig:
    # Model configs
    model_name: str = "unsloth/Llama-3.2-1B-Instruct"

    # Generation configs
    max_new_tokens: int = 500
    
    # Training configs
    batches_per_iteration: int = 10
    train_batch_size: int = 2
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 10
    
    # GRPO specific configs
    group_size: int = 8  # G in the paper
    epsilon: float = 0.2  # ε for clipping
    beta: float = 0.01   # KL penalty coefficient
    mu: int = 5         # Number of GRPO iterations per batch
    
    # Reward configs
    format_score: float = 0.1
    solve_score: float = 1.0
    
    # Logging configs
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500
    save_generations_every_n_steps: int = 1000
    generation_log_file: str = "generations.jsonl"

class CountdownGRPO:
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._init_model()
        
        self.train_dataset = get_dataset("train")
        self.test_dataset = get_dataset("test")
        
        wandb.init(
            project="countdown-grpo",
            config=vars(config),
            name=f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
            
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

    def _compute_kl_div(self, ref_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence using the unbiased estimator from the paper"""

        # TODO: This might be incorrect, not clear what it's vectorized over
        # Also, maybe this should be probs and not logits?
        ref_probs = torch.softmax(ref_logits, dim=-1)
        ratio = ref_probs / torch.softmax(logits, dim=-1)
        return (ratio - torch.log(ratio) - 1).mean()

    def _compute_log_probs(self, response_group: ResponseGroup) -> torch.Tensor:
        """Compute log probabilities for a list of responses using the current model."""

        # Need to iterate here for OOM safety
        log_probs_list = []
        for response_ids in response_group.response_token_ids:
            input_ids = response_ids.unsqueeze(0)
            outputs = self.model(input_ids, labels=input_ids)
            token_log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            log_probs_list.append(torch.gather(
                token_log_probs,
                dim=-1, 
                index=input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1).squeeze(0)[(response_group.generated_token_id_start-1):])

        return pad_sequence(log_probs_list, batch_first=True, padding_value=0)

    def _generate_response_group(
        self, 
        prompt: str,
        num_samples: int,
        max_new_tokens: int = 1000
    ) -> ResponseGroup:
        """Generate multiple responses and their log probs for a single prompt"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        responses = []
        response_token_ids = []
        log_probs = []
        
        #TODO: change to batch eval but watch padding and indexing
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                response_ids = outputs.sequences[0]
                responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
                
                # TODO: Check the indexing here, are we getting the right prob?
                scores = torch.stack(outputs.scores)  # [new_tokens, 1, vocab_size]
                token_log_probs = torch.log_softmax(scores.squeeze(1), dim=-1)  # [new_tokens, vocab_size]
                step_log_probs = torch.gather(
                    token_log_probs,
                    dim=-1,
                    index=response_ids[input_ids.shape[1]:].unsqueeze(-1)
                ).squeeze(-1)
                log_probs.append(step_log_probs)
                response_token_ids.append(response_ids)

        return ResponseGroup(responses=responses, response_token_ids=response_token_ids, log_probs_list=log_probs, generated_token_id_start=input_ids.shape[1], device=self.device)

    def _outer_iteration(self, optimizer: torch.optim.Optimizer) -> dict:
        """Perform one outer iteration of GRPO"""

        # this needs to be a deep copy
        # wait, no - if we get all training batches upfront, we can compute
        # ref probs for all of them! easy.
        reference_model = self.model

        for step in range(self.config.batches_per_iteration):
            training_batch = self.train_dataset.select(range(self.config.train_batch_size))

            # implicit iteration over elements of training_batch
            batch_response_groups = []
            batch_rewards = []
            batch_advantages = []
            # iteration over all prompts holding pi_theta_old model constant
            # potentially could relax this and let theta_old vary with task
            for task in training_batch:
                response_group = self._generate_response_group(
                    task["prompt"][0]["content"], 
                    self.config.group_size,
                    self.config.max_new_tokens,
                )

                batch_response_groups.append(response_group)
                rewards = self._compute_rewards(response_group.responses, task)
                batch_rewards.append(rewards)
                batch_advantages.append(self._normalize_advantages(rewards))
            
            for response_group, rewards, advantages in zip(batch_response_groups, batch_rewards, batch_advantages):
                for gradient_step in range(self.config.mu):
                    # Get log probs from current model for the same responses
                    # TODO: check if shape matches old_log_probs
                    # TODO: how to deal with fact that we're now sending full sequence? ask cursor

                    new_log_probs = self._compute_log_probs(response_group)
                    
                    # Compute probability ratios
                    ratio = torch.exp(new_log_probs - response_group.log_probs_tensor)
                    
                    # Compute clipped objective
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

                    # TODO: check if this is correct
                    policy_objective_by_response = policy_objective_matrix.multiply(response_group.sequence_mask).sum(dim=1).div(response_group.sequence_mask.sum(dim=1))
                    policy_objective = policy_objective_by_response.mean()

                    # Compute KL penalty
                    # This is supposed to be an average over responses and tokens, check if it's working
                    kl_div = self._compute_kl_div(response_group.log_probs_tensor, new_log_probs)
                    
                    # Total objective
                    objective = policy_objective - self.config.beta * kl_div
                    loss = -objective
                    
                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


        return {
            # tbd stuff for logging? is this the right interval?
        }
            
    @torch.no_grad()
    def evaluate(self, dataset: Dataset, num_examples: int = 100) -> dict:
        """Evaluate the model on a dataset"""
        total_rewards = []
        response_lengths = []
        
        for i, example in enumerate(dataset):
            if i >= num_examples:
                break
                
            prompt = example["prompt"][0]["content"]
            task = example["task"]
            
            # Generate single response for evaluation
            responses, _ = self._generate_group_responses(prompt, num_samples=1)
            response = responses[0]
            
            # Compute reward
            reward = self._compute_rewards([response], task)[0]
            
            total_rewards.append(reward.item())
            response_lengths.append(len(response))
            
        return {
            "mean_reward": np.mean(total_rewards),
            "mean_response_length": np.mean(response_lengths)
        }

    def train(self):
        """Main training loop"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        global_step = 0
        best_eval_reward = 0
        
        for epoch in range(self.config.num_epochs):
            for batch_idx, example in enumerate(self.train_dataset):
                prompt = example["prompt"][0]["content"]
                task = example["task"]
                
                # Perform GRPO update
                metrics = self._grpo_step(prompt, task, optimizer)
                global_step += 1
                
                # Log metrics
                if global_step % self.config.log_every_n_steps == 0:
                    logger.info(f"Step {global_step}: {metrics}")
                    if os.getenv("WANDB_API_KEY"):
                        wandb.log({
                            "train/loss": metrics["loss"],
                            "train/reward": metrics["rewards"],
                            "train/max_reward": metrics["max_reward"],
                            "train/mean_response_length": np.mean(metrics["response_lengths"]),
                            "global_step": global_step
                        })
                
                # Save sample generations
                if global_step % self.config.save_generations_every_n_steps == 0:
                    with open(self.config.generation_log_file, "a") as f:
                        json.dump({
                            "step": global_step,
                            "prompt": prompt,
                            "response": metrics["sample_response"]
                        }, f)
                        f.write("\n")
                
                # Evaluate
                if global_step % self.config.eval_every_n_steps == 0:
                    eval_metrics = self.evaluate(self.test_dataset)
                    logger.info(f"Evaluation: {eval_metrics}")
                    
                    if os.getenv("WANDB_API_KEY"):
                        wandb.log({
                            "eval/mean_reward": eval_metrics["mean_reward"],
                            "eval/mean_response_length": eval_metrics["mean_response_length"],
                            "global_step": global_step
                        })
                    
                    # Save best model
                    if eval_metrics["mean_reward"] > best_eval_reward:
                        best_eval_reward = eval_metrics["mean_reward"]
                        self.model.save_pretrained("best_model")
                        self.tokenizer.save_pretrained("best_model")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = GRPOConfig()
    trainer = CountdownGRPO(config)
    trainer.train()
