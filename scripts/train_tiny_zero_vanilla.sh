# Vanilla TinyZero training with fixed response length (no adaptive window)

python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=256 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.use_fused_kernels=True \
actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=72 \
actor_rollout_ref.actor.ppo_micro_batch_size=12 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=12 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.dtype=bfloat16 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=6 \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.model.fsdp_config.model_dtype=bfloat16 \
critic.ppo_mini_batch_size=72 \
critic.ppo_micro_batch_size=12 \
agent.adaptive_window.enable=False \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=100 \
trainer.project_name=TinyZero \
trainer.experiment_name=${EXPERIMENT_NAME}_vanilla \
trainer.total_epochs=15 2>&1 | tee verl_demo_vanilla.log


