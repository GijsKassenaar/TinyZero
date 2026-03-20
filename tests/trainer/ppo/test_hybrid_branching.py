import numpy as np
import torch

from verl import DataProto
from verl.trainer.config.algorithm import AlgoConfig
from verl.trainer.ppo.ray_trainer import (
    _attach_branch_tag,
    _concat_dataprotos_non_empty,
    _split_dataproto_by_mask,
    compute_hybrid_branch_advantages,
)


def _make_dataproto(batch_size: int, resp_len: int = 4) -> DataProto:
    responses = torch.arange(batch_size * resp_len, dtype=torch.long).reshape(batch_size, resp_len)
    token_rewards = torch.zeros((batch_size, resp_len), dtype=torch.float32)
    response_mask = torch.ones((batch_size, resp_len), dtype=torch.float32)
    exit_order = torch.ones((batch_size,), dtype=torch.long)

    non_tensors = {
        "uid": np.array([f"uid_{i}" for i in range(batch_size)], dtype=object),
    }

    return DataProto.from_dict(
        tensors={
            "responses": responses,
            "token_level_rewards": token_rewards,
            "response_mask": response_mask,
            "exit_order": exit_order,
        },
        non_tensors=non_tensors,
    )


def test_split_and_concat_preserves_uid_grouping():
    data = _make_dataproto(batch_size=6)
    data.non_tensor_batch["uid"] = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)

    mask = np.array([True, True, False, False, True, False], dtype=bool)
    left, right = _split_dataproto_by_mask(data, mask)

    _attach_branch_tag(left, tag_key="branch_mode", tag_value="sgrpo")
    _attach_branch_tag(right, tag_key="branch_mode", tag_value="grpo")

    mixed = _concat_dataprotos_non_empty([left, right])

    assert len(mixed) == len(data)
    assert set(mixed.non_tensor_batch["branch_mode"].tolist()) == {"sgrpo", "grpo"}

    # Group multiplicities should remain unchanged after split/reconcat.
    uid_vals, uid_counts = np.unique(mixed.non_tensor_batch["uid"], return_counts=True)
    uid_count_map = dict(zip(uid_vals.tolist(), uid_counts.tolist(), strict=True))
    assert uid_count_map == {"a": 2, "b": 2, "c": 2}


def test_mixed_advantages_cover_all_rows():
    batch_size = 6
    resp_len = 4
    data = _make_dataproto(batch_size=batch_size, resp_len=resp_len)

    # S-GRPO branch: 4 exits for uid_a.
    # GRPO branch: 2 full rollouts for uid_b.
    data.non_tensor_batch["uid"] = np.array(["uid_a", "uid_a", "uid_a", "uid_a", "uid_b", "uid_b"], dtype=object)
    data.non_tensor_batch["branch_mode"] = np.array(["sgrpo", "sgrpo", "sgrpo", "sgrpo", "grpo", "grpo"], dtype=object)

    data.batch["exit_order"] = torch.tensor([1, 2, 3, 4, 1, 1], dtype=torch.long)
    data.batch["token_level_rewards"] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out = compute_hybrid_branch_advantages(
        data,
        tag_key="branch_mode",
        gamma=1.0,
        lam=1.0,
        norm_adv_by_std_in_grpo=True,
        config=AlgoConfig(),
    )

    assert "advantages" in out.batch.keys()
    assert "returns" in out.batch.keys()
    assert out.batch["advantages"].shape == (batch_size, resp_len)
    assert out.batch["returns"].shape == (batch_size, resp_len)
    assert torch.isfinite(out.batch["advantages"]).all()
    assert torch.isfinite(out.batch["returns"]).all()


def test_branch_advantage_edge_cases_all_correct_or_all_incorrect():
    for tag in ("sgrpo", "grpo"):
        data = _make_dataproto(batch_size=4, resp_len=3)
        data.non_tensor_batch["uid"] = np.array(["u0", "u0", "u1", "u1"], dtype=object)
        data.non_tensor_batch["branch_mode"] = np.array([tag, tag, tag, tag], dtype=object)
        data.batch["exit_order"] = torch.tensor([1, 2, 1, 2], dtype=torch.long)
        data.batch["token_level_rewards"] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        out = compute_hybrid_branch_advantages(
            data,
            tag_key="branch_mode",
            gamma=1.0,
            lam=1.0,
            norm_adv_by_std_in_grpo=True,
            config=AlgoConfig(),
        )

        assert out.batch["advantages"].shape == data.batch["token_level_rewards"].shape
        assert out.batch["returns"].shape == data.batch["token_level_rewards"].shape
