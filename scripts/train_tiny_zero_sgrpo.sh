# S-GRPO TinyZero training: Serial-Group Decaying-Reward Policy Optimization
# This script trains with S-GRPO to encourage early correct answers and reduce overthinking.
#
# S-GRPO Two-Phase Generation:
# Phase 1: Generate ONE complete reasoning path per prompt (rollout.n=1)
# Phase 2: Create K-1 truncated versions by:
#   - Truncating at uniform positions in the reasoning
#   - Appending "answer inducer" text to force answer generation
#   - Generating continuations (answers) from truncated states
# Result: K samples per prompt (K-1 truncated + 1 full), rewards decay exponentially:
#   Exit 1 (earliest): correct = 1.0
#   Exit 2: correct = 0.5
#   Exit 3: correct = 0.25
#   Exit 4 (full): correct = 0.125
#
# Environment variables (set before running):
#   DATA_DIR: Directory containing train.parquet and test.parquet
#   BASE_MODEL: Path to base model (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
#   N_GPUS: Number of GPUs to use
#   ROLLOUT_TP_SIZE: Tensor parallel size for rollout (must divide N_GPUS)
#   EXPERIMENT_NAME: Name for the experiment (for logging)

python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=256 \
data.max_response_length=4096 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=72 \
actor_rollout_ref.actor.ppo_micro_batch_size=12 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=12 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.rollout.n=1 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_mini_batch_size=72 \
critic.ppo_micro_batch_size=12 \
algorithm.adv_estimator=sgrpo \
algorithm.sgrpo.enable=True \
algorithm.sgrpo.num_exits=4 \
algorithm.sgrpo.decay_factor=2.0 \
algorithm.sgrpo.exit_method=uniform \
algorithm.use_kl_in_reward=False \
trainer.logger=['wandb'] \
trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=500 \
trainer.test_freq=100 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo_sgrpo.log
