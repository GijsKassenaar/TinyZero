Please implement a hybrid branching mode in PPO trainer:
1. During active S-GRPO phase, generate one full response per prompt first.
2. Compute first-pass correctness.
3. For correct prompts: apply S-GRPO and generate 3 continuation exits from cutoffs.
4. For incorrect prompts: apply GRPO-style expansion by generating 3 additional full rollouts (so total 4 full responses including first pass).
5. Merge both branches into one batch for logprob/ref/value/reward stages.
6. Compute advantages per branch separately (SGRPO for correct branch, GRPO for incorrect branch), then concat.
7. Preserve uid grouping and add branch metrics.
8. Keep adaptive window fully compatible and unchanged by default.
9. Add config under algorithm.hybrid_branch and basic tests for shape/grouping correctness.

Implementation plan:

1. Add config for the hybrid branch mode in verl/trainer/config/algorithm.py and verl/trainer/config/ppo_trainer.yaml:
- algorithm.hybrid_branch.enable
- algorithm.hybrid_branch.correct_threshold default 0.5
- algorithm.hybrid_branch.incorrect_extra_rollouts default 3
- algorithm.hybrid_branch.tag_key default branch_mode

2. Add helper functions in verl/trainer/ppo/ray_trainer.py:
- build first-pass scoring batch and compute first-pass seq rewards
- split DataProto objects by correctness mask
- build incorrect GRPO expansion as first response plus 3 extra full rollouts
- merge branch outputs into one mixed batch with branch tags in non_tensor_batch

3. Rework generation section in verl/trainer/ppo/ray_trainer.py around the fit loop:
- still do initial full generation on all prompts
- if hybrid mode and sgrpo active:
  compute first-pass correctness
  correct branch uses S-GRPO controller
  incorrect branch uses full-rollout expansion
  concat branches and continue pipeline

4. Keep branch identity through balancing and reward:
- add non_tensor field branch_mode with values sgrpo or grpo
- ensure this survives repeat/concat/balance

5. Compute advantages per branch in verl/trainer/ppo/ray_trainer.py:
- select branch_mode == sgrpo -> compute_advantage with SGRPO
- select branch_mode == grpo -> compute_advantage with GRPO
- concat both back to one training batch

6. Add metrics:
- hybrid_branch/first_pass_correct_frac
- hybrid_branch/num_sgrpo_prompts
- hybrid_branch/num_grpo_prompts
- hybrid_branch/sgrpo_sample_frac
- hybrid_branch/grpo_sample_frac
- hybrid_branch/estimated_extra_generations

7. Keep adaptive window behavior unchanged:
- apply max_tokens to initial full pass and incorrect extra full rollouts
- leave S-GRPO continuation max_new_tokens behavior as-is unless you want to make it adaptive too

8. Add tests:
- unit test for mask split and recombination shape consistency
- unit test that uid grouping is preserved in both branches
- unit test that mixed advantage path produces advantages/returns for all rows
- smoke test with tiny batch where all correct and all incorrect edge cases