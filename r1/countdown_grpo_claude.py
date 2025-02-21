import os
import re
import torch
import wandb
from dataclasses import dataclass
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from datasets import Dataset
import numpy as np
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
import json
from datetime import datetime
from r1.data.countdown import get_dataset, compute_score, Task
from r1.data.core import Split, extract_task_response


@dataclass
class GRPOConfig:
    # Model configs
    model_name: str = "meta-llama/Llama-3.2-3B"
    max_length: int = 1000
    
    # Training configs
    batch_size: int = 4
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
        """Initialize the model and tokenizer with unsloth optimizations"""

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            dtype = "bfloat16",
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model = self.model.to(self.device)
        
    def _compute_rewards(self, responses: List[str], task: Task) -> torch.Tensor:
        """Compute rewards for a group of responses"""
        rewards = []
        # TODO: revise this: the compute_score already checks for rationale, so maybe
        # just adjust rationale scoring within the compute_score function
        # instead of this mess
        for response in responses:
            format_score = self.config.format_score if bool(re.search(r"<think>.*</think>", response)) else 0.0
            rewards.append(compute_score(
                response,
                task,
                format_score,
                self.config.solve_score
            ))
        return torch.tensor(rewards, device=self.device)

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

    def _generate_group_responses(
        self, 
        prompt: str,
        num_samples: int,
        max_new_tokens: int = 1000
    ) -> tuple[List[str], torch.Tensor]:
        """Generate multiple responses and their log probs for a single prompt"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        responses = []
        log_probs = []
        
        #TODO: change to batch eval
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
                responses.append(self.tokenizer.decode(response_ids[input_ids.shape[1]:], skip_special_tokens=True))
                
                # TODO: Check the indexing here, are we getting the right prob?
                scores = torch.stack(outputs.scores)  # [new_tokens, 1, vocab_size]
                token_log_probs = torch.log_softmax(scores.squeeze(1), dim=-1)  # [new_tokens, vocab_size]
                generated_token_ids = response_ids[input_ids.shape[1]:]
                step_log_probs = torch.gather(
                    token_log_probs,
                    dim=-1,
                    index=generated_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                log_probs.append(step_log_probs)
                
        # Pad log probs to same length
        log_probs = pad_sequence(log_probs, batch_first=True, padding_value=0)
        return responses, log_probs

# one grpo iteration
# store pi_current as pi_ref
# for steps 1..M
#  sample batch of B prompts
#  store pi_current as pi_old
#  sample G times for each prompt using pi_old
#  get rewards for G X B
#  advantage is normalized within G for one question
#  advantage is applied to all tokens within the response
#  (where did iteration over B go...?)
#  take mu steps:
#    compute ratio of current/old probs
#    compute policy loss
#    compute kl current vs ref
#    take a step

# question: does current keep changing with each of the mu steps? i would think yes.
# where are we iterating over B?

    def _grpo_step(
        self,
        prompt: str,
        task: Task,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """Perform one GRPO optimization step"""
        # Generate group of responses from current policy
        responses, old_log_probs = self._generate_group_responses(
            prompt, 
            self.config.group_size
        )
        
        # Compute rewards and advantages
        rewards = self._compute_rewards(responses, task)
        advantages = self._normalize_advantages(rewards)
        
        # Store initial model state for reference
        ref_state_dict = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        total_loss = 0
        # Multiple optimization iterations
        for _ in range(self.config.mu):
            # Generate new responses with current policy
            _, new_log_probs = self._generate_group_responses(
                prompt,
                self.config.group_size
            )
            
            # Compute probability ratios
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped objective
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.epsilon,
                1 + self.config.epsilon
            )
            
            # Compute losses
            policy_loss = -torch.min(
                ratio * advantages.unsqueeze(-1),
                clipped_ratio * advantages.unsqueeze(-1)
            ).mean()
            
            # Compute KL penalty
            kl_div = self._compute_kl_div(old_log_probs, new_log_probs)
            
            # Total loss
            loss = policy_loss + self.config.beta * kl_div
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return {
            "loss": total_loss / self.config.mu,
            "rewards": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "response_lengths": [len(r) for r in responses],
            "sample_response": responses[0]
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
    # Initialize and train
    config = GRPOConfig()
    trainer = CountdownGRPO(config)
    trainer.train()
