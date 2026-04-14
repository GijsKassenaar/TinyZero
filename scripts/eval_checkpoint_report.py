#!/usr/bin/env python
"""Evaluate saved TinyZero checkpoints via the training validation pipeline.

This script runs validation the same way training does (RayPPOTrainer._validate)
by launching `python -m verl.trainer.main_ppo` in val-only resume mode for each
checkpoint, one checkpoint at a time.

Key behavior:
- Uses checkpoint-specific training profiles (matching the checkpoint experiments).
- Limits evaluation set size by materializing a subset parquet (default: random 20 rows).
- Stores raw run logs and validation JSONL dumps per checkpoint.
- Produces summary CSV/JSON/Markdown report files.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_CHECKPOINTS = [
    {
        "name": "test_normalphased",
        "path": "test_normalphased/global_step_75",
        "profile": "normal_grpo",
    },
    {
        "name": "test_discounted_reasoning_0.999",
        "path": "test_discounted_reasoning_0.999/global_step_75",
        "profile": "discounted_reasoning",
    },
    {
        "name": "test_grpo_lambda_0.99",
        "path": "test_grpo_lambda_0.99/global_step_75",
        "profile": "grpo_lambda_base",
    },
    {
        "name": "test_grpo_lambda_sequence_gamma_discount",
        "path": "test_grpo_lambda_sequence_gamma_discount/global_step_75",
        "profile": "grpo_lambda_sequence_gamma_discount",
    },
    {
        "name": "test_grpo_lambda_0.99_token_normalization",
        "path": "test_grpo_lambda_0.99_token_normalization/global_step_75",
        "profile": "grpo_lambda_token_normalization_0p99",
    },
    {
        "name": "test_grpo_lambda_0.999_token_normalization",
        "path": "test_grpo_lambda_0.999_token_normalization/global_step_75",
        "profile": "grpo_lambda_token_normalization",
    },
    {
        "name": "test_grpo_lambda_second_trace",
        "path": "test_grpo_lambda_second_trace/global_step_75",
        "profile": "grpo_lambda_second_trace",
    },
]


PROFILE_OVERRIDES: dict[str, list[str]] = {
    "normal_grpo": [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
    ],
    "discounted_reasoning": [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.discounted_reasoning.enable=True",
        "algorithm.discounted_reasoning.gamma=0.999",
    ],
    "grpo_lambda_base": [
        "actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda",
        "critic.enable=False",
        "algorithm.adv_estimator=grpo_lambda",
        "algorithm.gamma=1.0",
        "algorithm.lam=0.99",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=False",
    ],
    "grpo_lambda_sequence_gamma_discount": [
        "actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda",
        "critic.enable=False",
        "algorithm.adv_estimator=grpo_lambda",
        "algorithm.gamma=1.0",
        "algorithm.lam=0.99",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.grpo_lambda_variant.enable=True",
        "algorithm.grpo_lambda_variant.flat_incorrect_trace=False",
        "algorithm.grpo_lambda_variant.sequence_gamma_discount_enable=True",
        "algorithm.grpo_lambda_variant.sequence_discount_gamma=0.99999999",
        "algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=True",
    ],
    "grpo_lambda_reasoning_only_0p99": [
        "actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda",
        "critic.enable=False",
        "algorithm.adv_estimator=grpo_lambda",
        "algorithm.gamma=1.0",
        "algorithm.lam=0.99",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.grpo_lambda_variant.enable=True",
        "algorithm.grpo_lambda_variant.flat_incorrect_trace=False",
        "algorithm.grpo_lambda_variant.sequence_gamma_discount_enable=True",
        "algorithm.grpo_lambda_variant.sequence_discount_gamma=0.99999999",
        "algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=True",
    ],
    "grpo_lambda_token_normalization": [
        "actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda",
        "critic.enable=False",
        "algorithm.adv_estimator=grpo_lambda",
        "algorithm.gamma=1.0",
        "algorithm.lam=0.999",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.grpo_lambda_variant.enable=True",
        "algorithm.grpo_lambda_variant.flat_incorrect_trace=False",
        "algorithm.grpo_lambda_variant.sequence_gamma_discount_enable=False",
        "algorithm.grpo_lambda_variant.sequence_discount_gamma=0.99999",
        "algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=False",
        "algorithm.grpo_lambda_variant.token_normalization_enable=True",
    ],
    "grpo_lambda_token_normalization_0p99": [
        "actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda",
        "critic.enable=False",
        "algorithm.adv_estimator=grpo_lambda",
        "algorithm.gamma=1.0",
        "algorithm.lam=0.99",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.grpo_lambda_variant.enable=True",
        "algorithm.grpo_lambda_variant.flat_incorrect_trace=False",
        "algorithm.grpo_lambda_variant.sequence_gamma_discount_enable=False",
        "algorithm.grpo_lambda_variant.sequence_discount_gamma=0.99999",
        "algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=False",
        "algorithm.grpo_lambda_variant.token_normalization_enable=True",
    ],
    "grpo_lambda_second_trace": [
        "actor_rollout_ref.actor.policy_loss.loss_mode=grpo_lambda",
        "critic.enable=False",
        "algorithm.adv_estimator=grpo_lambda",
        "algorithm.gamma=1.0",
        "algorithm.lam=0.999",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.grpo_lambda_variant.enable=True",
        "algorithm.grpo_lambda_variant.flat_incorrect_trace=False",
        "algorithm.grpo_lambda_variant.sequence_gamma_discount_enable=False",
        "algorithm.grpo_lambda_variant.sequence_discount_gamma=0.99999",
        "algorithm.grpo_lambda_variant.reasoning_only_discount_trace_enable=False",
        "algorithm.grpo_lambda_variant.token_normalization_enable=True",
        "algorithm.grpo_lambda_variant.second_trace_after_token_norm_enable=True",
        "algorithm.grpo_lambda_variant.second_trace_alpha=10",
    ],
}


@dataclass
class EvalResult:
    name: str
    profile: str
    checkpoint_path: str
    subset_rows: int
    return_code: int
    duration_sec: float
    mean_reward: Optional[float]
    accuracy: Optional[float]
    mean_response_length_correct: Optional[float]
    mean_response_length_incorrect: Optional[float]
    mean_reasoning_length_correct: Optional[float]
    mean_reasoning_length_incorrect: Optional[float]
    num_records: int
    status: str
    log_file: str
    generation_dump: str
    readable_dump: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trainer-style validation on checkpoints and build report.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help=(
            "Optional checkpoint paths. If omitted, uses the six built-in checkpoints. "
            "When paths are passed directly, profile is inferred from path name."
        ),
    )
    parser.add_argument(
        "--checkpoints_file",
        type=str,
        default=None,
        help="Optional text file with one checkpoint path per line.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing train.parquet and test.parquet.",
    )
    parser.add_argument("--split", type=str, default="test", help="Validation split name.")
    parser.add_argument(
        "--sample_count",
        type=int,
        default=20,
        help="Number of rows to keep from split parquet for validation.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="random",
        choices=["random", "head"],
        help="Subset selection mode: random sample or first N rows.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed used when sample_mode=random.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Base model path/name used to initialize architecture/tokenizer.",
    )
    parser.add_argument("--n_gpus", type=int, default=4, help="trainer.n_gpus_per_node value.")
    parser.add_argument(
        "--rollout_tp_size",
        type=int,
        default=1,
        help="actor_rollout_ref.rollout.tensor_model_parallel_size value.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="console",
        choices=["console", "wandb", "console_wandb"],
        help="Logger setting passed to trainer.logger.",
    )
    parser.add_argument("--project_name", type=str, default="TinyZero", help="trainer.project_name value.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/checkpoint_eval",
        help="Parent directory for all report artifacts.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name under output_dir.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess commands.",
    )
    parser.add_argument(
        "--append-run-dir",
        type=str,
        default=None,
        help=(
            "If set, append/update evaluated checkpoint rows in an existing checkpoint_eval run directory "
            "instead of creating a new run folder."
        ),
    )
    return parser.parse_args()


def logger_literal(logger_mode: str) -> str:
    if logger_mode == "console":
        return "['console']"
    if logger_mode == "wandb":
        return "['wandb']"
    return "['console','wandb']"


def infer_profile(path_text: str) -> str:
    lower = path_text.lower()
    if "lead_grpo_lambda" in lower:
        # Reuse the closest existing eval profile for lambda-based lead runs.
        return "grpo_lambda_base"
    if "lead_grpo" in lower:
        # Reuse normal GRPO eval profile for lead-GRPO runs.
        return "normal_grpo"
    if "discounted_reasoning" in lower:
        return "discounted_reasoning"
    if "reasoning_only" in lower and "grpo_lambda" in lower:
        if "0.99" in lower and "0.999" not in lower:
            return "grpo_lambda_reasoning_only_0p99"
        return "grpo_lambda_sequence_gamma_discount"
    if "0.999" in lower and "token_normalization" in lower:
        return "grpo_lambda_token_normalization"
    if "0.99_token_normalization" in lower:
        return "grpo_lambda_token_normalization_0p99"
    if "0.99" in lower and "token" in lower and "grpo_lambda" in lower and "0.999" not in lower:
        return "grpo_lambda_token_normalization_0p99"
    if "sequence_gamma_discount" in lower:
        return "grpo_lambda_sequence_gamma_discount"
    if "second_trace" in lower:
        return "grpo_lambda_second_trace"
    if "token_normalization" in lower:
        return "grpo_lambda_token_normalization"
    if "grpo_lambda_0.99" in lower:
        return "grpo_lambda_base"
    if "normalphased" in lower:
        return "normal_grpo"
    raise ValueError(f"Could not infer profile from checkpoint path: {path_text}")


def resolve_resume_checkpoint(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if (path / "actor").is_dir():
        resume_path = path
    elif path.name == "actor" and path.parent.name.startswith("global_step_"):
        resume_path = path.parent
    elif path.name == "huggingface" and path.parent.name == "actor" and path.parent.parent.name.startswith("global_step_"):
        resume_path = path.parent.parent
    else:
        raise ValueError(
            "Checkpoint path must be a global_step directory (or actor/huggingface within it). "
            f"Got: {path}"
        )

    actor_dir = resume_path / "actor"
    if not actor_dir.is_dir():
        raise ValueError(f"Checkpoint does not contain actor directory: {resume_path}")
    if not any(actor_dir.glob("model_world_size_*_rank_0.pt")):
        raise ValueError(f"Checkpoint actor directory is missing FSDP shard files: {actor_dir}")

    return resume_path


def build_checkpoint_items(args: argparse.Namespace) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    if args.checkpoints is None and args.checkpoints_file is None:
        return [dict(x) for x in DEFAULT_CHECKPOINTS]

    raw_paths: list[str] = []
    if args.checkpoints:
        raw_paths.extend(args.checkpoints)

    if args.checkpoints_file:
        lines = Path(args.checkpoints_file).expanduser().read_text().splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                raw_paths.append(line)

    for raw in raw_paths:
        profile = infer_profile(raw)
        name = Path(raw).name
        if name in ("actor", "huggingface"):
            name = Path(raw).parent.name
        elif name.startswith("global_step_"):
            parent_name = Path(raw).parent.name
            if parent_name:
                name = parent_name
        items.append({"name": name, "path": raw, "profile": profile})

    return items


def create_subset_parquet(
    data_dir: Path,
    split: str,
    sample_count: int,
    sample_mode: str,
    sample_seed: int,
    dst_path: Path,
) -> tuple[int, int]:
    src_path = data_dir / f"{split}.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing split parquet: {src_path}")

    df = pd.read_parquet(src_path)
    total_rows = len(df)
    keep_n = min(sample_count, total_rows)
    if sample_mode == "random":
        subset = df.sample(n=keep_n, random_state=sample_seed, replace=False)
    elif sample_mode == "head":
        subset = df.head(keep_n)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    subset = subset.reset_index(drop=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(dst_path, index=False)
    return len(subset), total_rows


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float, str]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    duration = time.time() - start
    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += "\n" + proc.stderr
    log_path.write_text(output)
    return proc.returncode, duration, output


def newest_jsonl_file(path: Path) -> Optional[Path]:
    files = sorted(path.glob("*.jsonl"))
    if not files:
        return None

    def sort_key(p: Path) -> tuple[int, str]:
        stem = p.stem
        return (int(stem), stem) if stem.isdigit() else (-1, stem)

    files = sorted(files, key=sort_key)
    return files[-1]


def mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def collapse_preview(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def extract_reasoning_text(output_text: str) -> str:
    if "</think>" in output_text:
        return output_text.split("</think>", 1)[0]
    return output_text


def load_checkpoint_tokenizer(resume_path: Path, base_model: str) -> tuple[Optional[Any], str]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"Could not import transformers: {exc}"

    candidates: list[tuple[str, bool]] = []
    local_hf = resume_path / "actor" / "huggingface"
    if local_hf.is_dir():
        candidates.append((str(local_hf), True))
    candidates.append((base_model, False))

    errors: list[str] = []
    for source, local_only in candidates:
        kwargs = {"trust_remote_code": True}
        if local_only:
            kwargs["local_files_only"] = True
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, fix_mistral_regex=True, **kwargs)
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            return tokenizer, ""
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{source}: {exc}")

    return None, "; ".join(errors)


def parse_generation_dump(
    jsonl_path: Path,
    tokenizer: Optional[Any],
    readable_dump_path: Optional[Path] = None,
) -> dict[str, Any]:
    scores: list[float] = []
    response_len_correct: list[float] = []
    response_len_incorrect: list[float] = []
    reasoning_len_correct: list[float] = []
    reasoning_len_incorrect: list[float] = []
    readable_rows: list[dict[str, Any]] = []

    with jsonl_path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            score: Optional[float] = None
            if "score" in row:
                score = float(row["score"])
            elif "reward" in row:
                score = float(row["reward"])

            if score is not None:
                scores.append(score)

            output_raw = row.get("output", "")
            if isinstance(output_raw, str):
                output_text = output_raw
            elif output_raw is None:
                output_text = ""
            else:
                output_text = str(output_raw)

            input_raw = row.get("input", "")
            if isinstance(input_raw, str):
                input_text = input_raw
            elif input_raw is None:
                input_text = ""
            else:
                input_text = str(input_raw)

            is_correct = bool(score is not None and score >= 0.99)
            response_len_tokens: Optional[int] = None
            reasoning_len_tokens: Optional[int] = None
            if tokenizer is not None:
                response_len_tokens = int(len(tokenizer.encode(output_text, add_special_tokens=False)))
                reasoning_text = extract_reasoning_text(output_text)
                reasoning_len_tokens = int(len(tokenizer.encode(reasoning_text, add_special_tokens=False)))

                if is_correct:
                    response_len_correct.append(float(response_len_tokens))
                    reasoning_len_correct.append(float(reasoning_len_tokens))
                else:
                    response_len_incorrect.append(float(response_len_tokens))
                    reasoning_len_incorrect.append(float(reasoning_len_tokens))

            readable_rows.append(
                {
                    "idx": idx,
                    "step": row.get("step"),
                    "score": score,
                    "is_correct": is_correct,
                    "response_len_tokens": response_len_tokens,
                    "reasoning_len_tokens": reasoning_len_tokens,
                    "input_preview": collapse_preview(input_text),
                    "output_preview": collapse_preview(output_text),
                }
            )

    readable_path_text = ""
    if readable_dump_path is not None:
        readable_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with readable_dump_path.open("w") as f:
            for row in readable_rows:
                f.write(json.dumps(row) + "\n")
        readable_path_text = str(readable_dump_path)

    if not scores:
        return {
            "mean_reward": None,
            "accuracy": None,
            "num_records": 0,
            "mean_response_length_correct": mean_or_none(response_len_correct),
            "mean_response_length_incorrect": mean_or_none(response_len_incorrect),
            "mean_reasoning_length_correct": mean_or_none(reasoning_len_correct),
            "mean_reasoning_length_incorrect": mean_or_none(reasoning_len_incorrect),
            "readable_dump": readable_path_text,
        }

    mean_reward = float(sum(scores) / len(scores))
    accuracy = float(sum(1 for x in scores if x >= 0.99) / len(scores))
    return {
        "mean_reward": mean_reward,
        "accuracy": accuracy,
        "num_records": len(scores),
        "mean_response_length_correct": mean_or_none(response_len_correct),
        "mean_response_length_incorrect": mean_or_none(response_len_incorrect),
        "mean_reasoning_length_correct": mean_or_none(reasoning_len_correct),
        "mean_reasoning_length_incorrect": mean_or_none(reasoning_len_incorrect),
        "readable_dump": readable_path_text,
    }


def markdown_float(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.4f}"


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def load_summary_rows(summary_csv: Path) -> tuple[list[str], list[dict[str, object]]]:
    if not summary_csv.exists():
        return [], []
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)
    return fieldnames, rows


def merge_existing_and_new_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    def row_key(row: dict[str, object]) -> str:
        checkpoint_path = str(row.get("checkpoint_path", "")).strip()
        if checkpoint_path:
            return f"cp::{checkpoint_path}"
        name = str(row.get("name", "")).strip()
        return f"name::{name}"

    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []

    for row in existing_rows:
        key = row_key(row)
        if key in ("cp::", "name::"):
            continue
        if key not in merged:
            order.append(key)
            merged[key] = {}
        merged[key].update(row)

    for row in new_rows:
        key = row_key(row)
        if key in ("cp::", "name::"):
            continue
        if key not in merged:
            order.append(key)
            merged[key] = {}
        merged[key].update(row)

    return [merged[key] for key in order]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser().resolve()

    checkpoints = build_checkpoint_items(args)
    if len(checkpoints) == 0:
        print("No checkpoints provided.")
        return 1

    if args.append_run_dir:
        run_dir = Path(args.append_run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"append-run-dir does not exist: {run_dir}")
        run_name = run_dir.name
    else:
        run_name = args.run_name or f"checkpoint_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(args.output_dir).expanduser() / run_name
    logs_dir = run_dir / "logs"
    subset_dir = run_dir / "subsets"
    validation_dump_dir = run_dir / "validation_data"
    trainer_local_dir = run_dir / "trainer_local"

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    validation_dump_dir.mkdir(parents=True, exist_ok=True)
    trainer_local_dir.mkdir(parents=True, exist_ok=True)

    subset_path = subset_dir / f"{args.split}_{args.sample_mode}_{args.sample_count}.parquet"
    subset_rows, total_rows = create_subset_parquet(
        data_dir=data_dir,
        split=args.split,
        sample_count=args.sample_count,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        dst_path=subset_path,
    )

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}
    else:
        metadata = {}

    metadata.update(
        {
            "run_name": run_name,
            "repo_root": str(repo_root),
            "data_dir": str(data_dir),
            "split": args.split,
            "sample_count_requested": args.sample_count,
            "sample_count_written": subset_rows,
            "sample_mode": args.sample_mode,
            "sample_seed": args.sample_seed,
            "split_total_rows": total_rows,
            "subset_path": str(subset_path),
            "last_update_at": datetime.now().isoformat(timespec="seconds"),
            "last_update_args": vars(args),
            "last_update_checkpoints": checkpoints,
        }
    )
    if "created_at" not in metadata:
        metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger_value = logger_literal(args.logger)
    results: list[EvalResult] = []

    for idx, ckpt in enumerate(checkpoints, start=1):
        name = ckpt["name"]
        profile = ckpt["profile"]
        raw_path = ckpt["path"]

        if profile not in PROFILE_OVERRIDES:
            results.append(
                EvalResult(
                    name=name,
                    profile=profile,
                    checkpoint_path=raw_path,
                    subset_rows=subset_rows,
                    return_code=-1,
                    duration_sec=0.0,
                    mean_reward=None,
                    accuracy=None,
                    mean_response_length_correct=None,
                    mean_response_length_incorrect=None,
                    mean_reasoning_length_correct=None,
                    mean_reasoning_length_incorrect=None,
                    num_records=0,
                    status="failed",
                    log_file=str(logs_dir / f"{name}.validate.log"),
                    generation_dump="",
                    readable_dump="",
                    error=f"Unknown profile: {profile}",
                )
            )
            continue

        log_path = logs_dir / f"{name}.validate.log"
        dump_dir = validation_dump_dir / name
        dump_dir.mkdir(parents=True, exist_ok=True)
        local_dir = trainer_local_dir / name
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(checkpoints)}] Validating {name} ({profile})")

        try:
            resume_path = resolve_resume_checkpoint(raw_path, repo_root)

            common_overrides = [
                f"data.train_files={data_dir / 'train.parquet'}",
                f"data.val_files={subset_path}",
                "data.train_batch_size=256",
                "data.val_batch_size=256",
                "data.dataloader_num_workers=2",
                "data.max_prompt_length=256",
                "data.max_response_length=2048",
                f"actor_rollout_ref.model.path={args.base_model}",
                "actor_rollout_ref.model.use_remove_padding=True",
                "actor_rollout_ref.actor.use_dynamic_bsz=True",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.ppo_mini_batch_size=32",
                "actor_rollout_ref.actor.ppo_micro_batch_size=8",
                "actor_rollout_ref.rollout.n=4",
                f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp_size}",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size=32",
                "actor_rollout_ref.rollout.val_kwargs.n=1",
                "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
                "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
                "actor_rollout_ref.ref.log_prob_micro_batch_size=8",
                "agent.entropy_logging.enable=False",
                "agent.adaptive_window.enable=True",
                "agent.adaptive_window.mode=phased",
                "agent.adaptive_window.initial_window=1024",
                "agent.adaptive_window.max_window=2048",
                "agent.adaptive_window.phased_schedule=[[50,2048]]",
                f"trainer.n_gpus_per_node={args.n_gpus}",
                "trainer.nnodes=1",
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={resume_path}",
                "trainer.total_training_steps=75",
                "trainer.save_freq=-1",
                "trainer.test_freq=0",
                "trainer.val_before_train=True",
                "trainer.val_only=True",
                f"trainer.logger={logger_value}",
                f"trainer.project_name={args.project_name}",
                f"trainer.experiment_name={run_name}__{name}",
                f"trainer.default_local_dir={local_dir}",
                f"trainer.validation_data_dir={dump_dir}",
            ]

            cmd = [args.python, "-m", "verl.trainer.main_ppo"]
            cmd.extend(common_overrides)
            cmd.extend(PROFILE_OVERRIDES[profile])

            return_code, duration_sec, _ = run_command(cmd, repo_root, log_path)

            if return_code != 0:
                results.append(
                    EvalResult(
                        name=name,
                        profile=profile,
                        checkpoint_path=str(resume_path),
                        subset_rows=subset_rows,
                        return_code=return_code,
                        duration_sec=duration_sec,
                        mean_reward=None,
                        accuracy=None,
                        mean_response_length_correct=None,
                        mean_response_length_incorrect=None,
                        mean_reasoning_length_correct=None,
                        mean_reasoning_length_incorrect=None,
                        num_records=0,
                        status="failed",
                        log_file=str(log_path),
                        generation_dump="",
                        readable_dump="",
                        error="Validation command failed. See log file.",
                    )
                )
                continue

            dump_file = newest_jsonl_file(dump_dir)
            if dump_file is None:
                results.append(
                    EvalResult(
                        name=name,
                        profile=profile,
                        checkpoint_path=str(resume_path),
                        subset_rows=subset_rows,
                        return_code=return_code,
                        duration_sec=duration_sec,
                        mean_reward=None,
                        accuracy=None,
                        mean_response_length_correct=None,
                        mean_response_length_incorrect=None,
                        mean_reasoning_length_correct=None,
                        mean_reasoning_length_incorrect=None,
                        num_records=0,
                        status="failed",
                        log_file=str(log_path),
                        generation_dump="",
                        readable_dump="",
                        error="No validation JSONL dump found.",
                    )
                )
                continue

            tokenizer, tokenizer_error = load_checkpoint_tokenizer(resume_path, args.base_model)
            readable_dump_path = dump_file.with_suffix(".readable.jsonl")
            parsed = parse_generation_dump(
                dump_file,
                tokenizer=tokenizer,
                readable_dump_path=readable_dump_path,
            )
            mean_reward = parsed["mean_reward"]
            accuracy = parsed["accuracy"]
            num_records = int(parsed["num_records"])
            mean_response_length_correct = parsed["mean_response_length_correct"]
            mean_response_length_incorrect = parsed["mean_response_length_incorrect"]
            mean_reasoning_length_correct = parsed["mean_reasoning_length_correct"]
            mean_reasoning_length_incorrect = parsed["mean_reasoning_length_incorrect"]
            readable_dump = str(parsed["readable_dump"])

            status = "ok" if mean_reward is not None and accuracy is not None else "ok_no_scores"
            error_parts: list[str] = []
            if status != "ok":
                error_parts.append("Validation completed but no score fields found in dump.")
            if tokenizer is None:
                error_parts.append(f"Tokenizer unavailable for length metrics: {tokenizer_error}")
            error = " | ".join(error_parts)

            results.append(
                EvalResult(
                    name=name,
                    profile=profile,
                    checkpoint_path=str(resume_path),
                    subset_rows=subset_rows,
                    return_code=return_code,
                    duration_sec=duration_sec,
                    mean_reward=mean_reward,
                    accuracy=accuracy,
                    mean_response_length_correct=mean_response_length_correct,
                    mean_response_length_incorrect=mean_response_length_incorrect,
                    mean_reasoning_length_correct=mean_reasoning_length_correct,
                    mean_reasoning_length_incorrect=mean_reasoning_length_incorrect,
                    num_records=num_records,
                    status=status,
                    log_file=str(log_path),
                    generation_dump=str(dump_file),
                    readable_dump=readable_dump,
                    error=error,
                )
            )

        except Exception as exc:  # pylint: disable=broad-except
            results.append(
                EvalResult(
                    name=name,
                    profile=profile,
                    checkpoint_path=raw_path,
                    subset_rows=subset_rows,
                    return_code=-1,
                    duration_sec=0.0,
                    mean_reward=None,
                    accuracy=None,
                    mean_response_length_correct=None,
                    mean_response_length_incorrect=None,
                    mean_reasoning_length_correct=None,
                    mean_reasoning_length_incorrect=None,
                    num_records=0,
                    status="failed",
                    log_file=str(log_path),
                    generation_dump="",
                    readable_dump="",
                    error=str(exc),
                )
            )

    new_rows = [asdict(r) for r in results]

    csv_path = run_dir / "summary.csv"
    existing_fieldnames, existing_rows = load_summary_rows(csv_path)
    merged_rows = merge_existing_and_new_rows(existing_rows, new_rows)

    fieldnames = list(existing_fieldnames)
    if not fieldnames and len(new_rows) > 0:
        fieldnames = list(new_rows[0].keys())
    for row in new_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            out_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out_row)

    json_path = run_dir / "summary.json"
    json_path.write_text(json.dumps(merged_rows, indent=2))

    md_lines = [
        "# Checkpoint Validation Report",
        "",
        f"- Run: {run_name}",
        f"- Created: {datetime.now().isoformat(timespec='seconds')}",
        f"- Validation mode: trainer val-only (same pipeline as training _validate)",
        f"- Split: {args.split}",
        f"- Requested sample count: {args.sample_count}",
        f"- Actual sample count: {subset_rows}",
        f"- Sample mode: {args.sample_mode}",
        f"- Sample seed: {args.sample_seed}",
        f"- Subset parquet: {subset_path}",
        "",
        "| Name | Profile | Mean Reward | Accuracy | Resp Len (C) | Resp Len (I) | Reason Len (C) | Reason Len (I) | Records | Status | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]

    for row in merged_rows:
        mean_reward = _to_float(row.get("mean_reward"))
        accuracy = _to_float(row.get("accuracy"))
        resp_len_correct = _to_float(row.get("mean_response_length_correct"))
        resp_len_incorrect = _to_float(row.get("mean_response_length_incorrect"))
        reason_len_correct = _to_float(row.get("mean_reasoning_length_correct"))
        reason_len_incorrect = _to_float(row.get("mean_reasoning_length_incorrect"))
        num_records = _to_int(row.get("num_records"))
        duration_sec = _to_float(row.get("duration_sec"))
        md_lines.append(
            f"| {row.get('name', '')} | {row.get('profile', '')} | {markdown_float(mean_reward)} | {markdown_float(accuracy)} "
            + f"| {markdown_float(resp_len_correct)} | {markdown_float(resp_len_incorrect)} "
            + f"| {markdown_float(reason_len_correct)} | {markdown_float(reason_len_incorrect)} "
            + f"| {num_records if num_records is not None else 'N/A'} | {row.get('status', '')} | {markdown_float(duration_sec)} |"
        )

    md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append("")
    md_lines.append(f"- Summary CSV: {csv_path}")
    md_lines.append(f"- Summary JSON: {json_path}")
    md_lines.append(f"- Logs directory: {logs_dir}")
    md_lines.append(f"- Validation dumps: {validation_dump_dir}")

    failures = [row for row in merged_rows if str(row.get("status", "")) != "ok"]
    if failures:
        md_lines.append("")
        md_lines.append("## Failures")
        md_lines.append("")
        for row in failures:
            md_lines.append(f"- {row.get('name', '')}: {row.get('error', '')}")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(md_lines) + "\n")

    print("Done.")
    print(f"Report: {report_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Subset parquet: {subset_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
