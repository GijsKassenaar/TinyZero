# Adaptive TinyZero training with dynamic response window

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
actor_rollout_ref.rollout.dtype=bfloat16 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.actor.entropy_coeff=0 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.model.fsdp_config.model_dtype=bfloat16 \
critic.ppo_mini_batch_size=72 \
critic.ppo_micro_batch_size=12 \
agent.adaptive_window.enable=True \
agent.adaptive_window.initial_window=512 \
agent.adaptive_window.min_window=20 \
agent.adaptive_window.max_window=4096 \
agent.adaptive_window.warmup_steps=0 \
agent.adaptive_window.std_multiplier=0.5 \
agent.adaptive_window.success_threshold=0.1 \
agent.adaptive_window.min_samples=2 \
agent.adaptive_window.update_frequency=3 \
agent.adaptive_window.epsilon=0.0 \
algorithm.cosine_reward.enable=False \
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
trainer.total_epochs=15 2>&1 | tee verl_demo_adaptive.log



