import numpy as np
import torch

from verl.trainer.config.algorithm import AlgoConfig, GRPOLambdaVariantConfig
from verl.trainer.ppo.core_algos import compute_grpo_lambda_outcome_advantage


def _make_inputs(scale: float = 1.0):
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ) * scale
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = np.array(["uid_a", "uid_a", "uid_a"], dtype=object)
    return token_level_rewards, response_mask, index


def _cfg(
    carry: bool,
    additive_normalization_enable: bool = False,
    additive_normalization_tau: float = 0.02,
    reasoning_only_discount_trace_enable: bool = False,
    second_trace_after_token_norm_enable: bool = False,
    second_trace_alpha: float = 1.0,
) -> AlgoConfig:
    return AlgoConfig(
        grpo_lambda_variant=GRPOLambdaVariantConfig(
            enable=True,
            reasoning_only_discount_trace_enable=reasoning_only_discount_trace_enable,
            token_normalization_enable=carry,
            additive_normalization_enable=additive_normalization_enable,
            additive_normalization_tau=additive_normalization_tau,
            second_trace_after_token_norm_enable=second_trace_after_token_norm_enable,
            second_trace_alpha=second_trace_alpha,
            correctness_threshold=0.5,
        )
    )


def test_token_normalization_changes_token_advantages():
    rewards, mask, index = _make_inputs(scale=1.0)

    adv_baseline, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=False),
    )

    adv_carry, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True),
    )

    assert adv_baseline.shape == adv_carry.shape
    assert torch.isfinite(adv_baseline).all()
    assert torch.isfinite(adv_carry).all()

    valid = mask.bool()
    max_abs_diff = torch.max(torch.abs((adv_baseline - adv_carry)[valid]))
    assert max_abs_diff > 1e-5


def test_token_normalization_scaling_by_norm_mode():
    rewards, mask, index = _make_inputs(scale=1.0)
    rewards_scaled, _, _ = _make_inputs(scale=3.0)

    adv_z, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True),
    )
    adv_z_scaled, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards_scaled,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True),
    )

    adv_centered, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=False,
        config=_cfg(carry=True),
    )
    adv_centered_scaled, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards_scaled,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=False,
        config=_cfg(carry=True),
    )

    valid = mask.bool()
    assert torch.allclose(adv_z[valid], adv_z_scaled[valid], atol=1e-5, rtol=1e-4)
    assert torch.allclose(adv_centered_scaled[valid], 3.0 * adv_centered[valid], atol=1e-6, rtol=1e-5)


def test_default_path_matches_when_carry_disabled():
    rewards, mask, index = _make_inputs(scale=1.0)

    adv_no_variant, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.9,
        norm_adv_by_std_in_grpo=True,
        config=None,
    )

    adv_variant_no_carry, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.9,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=False),
    )

    assert torch.allclose(adv_no_variant, adv_variant_no_carry, atol=1e-7, rtol=1e-7)


def test_token_normalization_traces_before_normalizing():
    # Sparse terminal-only outcomes should still produce non-zero early-token advantages
    # when terminal reward is first traced backward and then normalized.
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = np.array(["uid_a", "uid_a", "uid_a"], dtype=object)

    adv, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.9,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True),
    )

    valid_t0 = mask[:, 0].bool()
    assert torch.max(torch.abs(adv[valid_t0, 0])) > 1e-6


def test_token_normalization_additive_tau_changes_scale_behavior():
    rewards, mask, index = _make_inputs(scale=1.0)
    rewards_scaled, _, _ = _make_inputs(scale=3.0)

    adv_add, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True, additive_normalization_enable=True, additive_normalization_tau=0.02),
    )
    adv_add_scaled, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards_scaled,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True, additive_normalization_enable=True, additive_normalization_tau=0.02),
    )

    valid = mask.bool()
    # Unlike z-score normalization, additive denominator (std + tau) is not scale-invariant.
    assert torch.max(torch.abs((adv_add - adv_add_scaled)[valid])) > 1e-4


def test_reasoning_only_trace_requires_reasoning_mask():
    rewards, mask, index = _make_inputs(scale=1.0)

    try:
        compute_grpo_lambda_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            gamma=1.0,
            lam=0.5,
            norm_adv_by_std_in_grpo=False,
            config=_cfg(carry=False, reasoning_only_discount_trace_enable=True),
            reasoning_token_mask=None,
        )
        assert False, "Expected ValueError when reasoning_only_discount_trace_enable=True without reasoning_token_mask"
    except ValueError as e:
        assert "reasoning_token_mask must be provided" in str(e)


def test_reasoning_only_trace_applies_decay_inside_reasoning_tokens_only():
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones_like(rewards)
    reasoning_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = np.array(["uid_a", "uid_a"], dtype=object)

    adv, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=False,
        config=_cfg(carry=False, reasoning_only_discount_trace_enable=True),
        reasoning_token_mask=reasoning_mask,
    )

    expected = torch.tensor(
        [
            [0.25, 0.5, 0.5, 0.5],
            [-0.25, -0.5, -0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(adv, expected, atol=1e-6, rtol=1e-6)


def test_second_trace_after_token_norm_changes_advantages():
    rewards, mask, index = _make_inputs(scale=1.0)

    adv_base, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(carry=True, second_trace_after_token_norm_enable=False),
    )

    adv_second_trace, _ = compute_grpo_lambda_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=0.5,
        norm_adv_by_std_in_grpo=True,
        config=_cfg(
            carry=True,
            second_trace_after_token_norm_enable=True,
            second_trace_alpha=2.0,
        ),
    )

    valid = mask.bool()
    assert torch.max(torch.abs((adv_base - adv_second_trace)[valid])) > 1e-5
