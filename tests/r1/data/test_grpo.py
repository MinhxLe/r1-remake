import torch
from scripts.train_grpo_countdown import GRPOTrainer, Cfg, Group


# Constants for test cases
EPSILON = 0.2  # Same as default in Cfg
BETA = 0.001  # Same as default in Cfg


class TestComputeNormalizedAdvantages:
    def test_basic_normalization(self):
        rewards = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32)
        result = GRPOTrainer._compute_normalized_advantages(rewards)
        torch.testing.assert_close(result, expected)

    def test_single_value(self):
        rewards = torch.tensor([1.0], dtype=torch.float32)
        expected = torch.tensor([0.0], dtype=torch.float32)
        result = GRPOTrainer._compute_normalized_advantages(rewards)
        torch.testing.assert_close(result, expected)


class TestComputePolicyObjective:
    def setup_method(self):
        self.cfg = Cfg()
        self.trainer = GRPOTrainer(cfg=self.cfg, train_dataset=[], test_dataset=[])

    def test_equal_log_probs(self):
        new_log_probs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)
        old_log_probs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)
        advantages = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

        result = self.trainer._compute_policy_objective(
            new_log_probs, old_log_probs, advantages
        )
        torch.testing.assert_close(result, advantages)

    def test_clipped_ratio(self):
        new_log_probs = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        old_log_probs = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        advantages = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

        expected_clip = torch.ones_like(advantages) * (1 + EPSILON)
        result = self.trainer._compute_policy_objective(
            new_log_probs, old_log_probs, advantages
        )
        torch.testing.assert_close(result, expected_clip * advantages)


class TestComputeKLDivergence:
    def test_identical_distributions(self):
        ref_log_probs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)
        model_log_probs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)
        expected = torch.zeros_like(ref_log_probs)

        result = GRPOTrainer._compute_kl_divergence(ref_log_probs, model_log_probs)
        torch.testing.assert_close(result, expected)

    def test_different_distributions(self):
        ref_log_probs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)
        model_log_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

        result = GRPOTrainer._compute_kl_divergence(ref_log_probs, model_log_probs)
        assert (result >= 0).all()


class TestComputeTotalObjective:
    def setup_method(self):
        self.cfg = Cfg()
        self.trainer = GRPOTrainer(cfg=self.cfg, train_dataset=[], test_dataset=[])

    def test_masked_objective(self):
        policy_objectives = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        kls = torch.tensor([[0.1, 0.1], [0.1, 0.0]], dtype=torch.float32)
        response_masks = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)

        group = Group(
            prompt_token_ids=torch.tensor([]),
            response_token_ids=torch.tensor([]),
            response_masks=response_masks,
            response_log_probs=torch.tensor([]),
        )

        expected = torch.tensor(1 - 0.1 * BETA)
        result = self.trainer._compute_total_objective(policy_objectives, kls, group)
        torch.testing.assert_close(result, expected)

    def test_all_masked(self):
        policy_objectives = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        kls = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
        response_masks = torch.tensor([[1, 1]], dtype=torch.float32)

        group = Group(
            prompt_token_ids=torch.tensor([]),
            response_token_ids=torch.tensor([]),
            response_masks=response_masks,
            response_log_probs=torch.tensor([]),
        )

        expected = torch.tensor(1 - 0.1 * BETA)
        result = self.trainer._compute_total_objective(policy_objectives, kls, group)
        torch.testing.assert_close(result, expected)
