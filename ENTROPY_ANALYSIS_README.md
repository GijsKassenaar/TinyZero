# Entropy Analysis Scripts

## Overview

Two scripts for analyzing entropy and varentropy data from GRPO training:

1. **`entropy_analysis.py`** - Single file analysis (original)
2. **`entropy_analysis_batch.py`** - Batch analysis comparing correct vs incorrect answers (NEW)

## Getting Entropy Data from Snellius

After your training run completes, download the entropy data:

```bash
# Download entire entropy data directory
scp -r gkassenaar@snellius.surf.nl:~/TinyZero/checkpoints/TinyZero/entropy_test/entropy_data/ ./

# Or download specific files
scp gkassenaar@snellius.surf.nl:~/TinyZero/checkpoints/TinyZero/entropy_test/entropy_data/entropy_step_*.pt ./entropy_data/
```

## Usage

### Batch Analysis (Recommended)

Analyzes all entropy files at once and compares correct vs incorrect answers:

```bash
python entropy_analysis_batch.py checkpoints/TinyZero/entropy_test/entropy_data/
```

This will:
- Load all `entropy_step_*.pt` files
- Separate samples by correct/incorrect answers
- Generate 4 comprehensive comparison plots
- Print detailed statistics

### Output

Creates an `entropy_analysis/` directory with:

1. **`success_rate_over_time.png`**
   - Shows how success rate evolves during training
   
2. **`entropy_varentropy_comparison.png`**
   - Mean entropy and varentropy for correct vs incorrect answers across steps
   
3. **`positional_comparison.png`**
   - How entropy/varentropy change by token position in the response
   - Averaged across all steps
   - Separate curves for correct vs incorrect
   
4. **`distributions_combined.png`**
   - Histograms and scatter plots showing distributions
   - Combined data from all steps

### Single File Analysis (Original)

For analyzing one specific step:

```bash
python entropy_analysis.py
# Edit line 6 to change which file to load
```

## What the Metrics Mean

### **Entropy** (Shannon Entropy)
- Measures uncertainty in the model's predictions
- **Lower entropy** = Model is more confident
- **Higher entropy** = Model is uncertain

**Hypothesis:** Correct answers might have lower entropy (model is more confident)

### **Varentropy** (Variance of Entropy)
- Measures how much entropy varies across the probability distribution
- **Lower varentropy** = Uniform confidence across tokens
- **Higher varentropy** = Some tokens very certain, others very uncertain

**Hypothesis:** Incorrect answers might have higher varentropy (inconsistent confidence)

## Understanding the Results

### Expected Patterns for Countdown Task

**Correct answers might show:**
- Lower mean entropy (model confident)
- Lower varentropy (consistent confidence)
- Entropy decreases toward end of response (more certain as solution progresses)

**Incorrect answers might show:**
- Higher mean entropy (model uncertain)
- Higher varentropy (inconsistent confidence)
- Entropy remains high throughout (model confused)

### Positional Patterns

The positional analysis shows entropy/varentropy at each token position:

```
Position 0-50:   Initial thinking (<think> section)
Position 50-200: Main computation (countdown steps)
Position 200+:   Final answer (<answer> section)
```

Look for:
- Where do correct vs incorrect diverge?
- Does entropy spike at specific positions?
- Does the model become more/less confident over time?

## Advanced Analysis

### Custom Analysis

You can load the data yourself for custom analysis:

```python
import torch
from pathlib import Path

# Load all files
data_dir = Path('entropy_data/')
all_data = []
for file in sorted(data_dir.glob('entropy_step_*.pt')):
    data = torch.load(file)
    all_data.append(data)

# Each file contains:
# - old_entropy: (batch_size, response_len) - entropy per token
# - old_varentropy: (batch_size, response_len) - varentropy per token
# - attention_mask: (batch_size, total_len) - valid token mask
# - rewards: (batch_size,) - 1.0 for correct, 0.0 for incorrect
# - responses: (batch_size, response_len) - token IDs

# Your analysis here...
```

### Statistical Tests

Add statistical significance testing:

```python
from scipy import stats

correct_entropy = all_correct_entropy.numpy()
incorrect_entropy = all_incorrect_entropy.numpy()

# T-test
t_stat, p_value = stats.ttest_ind(correct_entropy, incorrect_entropy)
print(f"T-test: t={t_stat:.4f}, p={p_value:.4e}")

# Mann-Whitney U test (non-parametric)
u_stat, p_value = stats.mannwhitneyu(correct_entropy, incorrect_entropy)
print(f"Mann-Whitney U: U={u_stat:.4f}, p={p_value:.4e}")
```

## Troubleshooting

### "No entropy files found"
- Check the path to entropy_data directory
- Ensure files are named `entropy_step_*.pt`
- Files are only generated if `agent.entropy_logging.enable=True` in config

### "No reward data available"
- Rewards are only saved during training, not validation
- Check that your run actually computed rewards
- Early test runs might not have reward data

### Memory Issues
If loading all files causes OOM:
- Process files one at a time
- Reduce the number of steps analyzed
- Use the single-file analysis script instead

## Questions to Answer

Using these plots, you can investigate:

1. **Do correct answers have systematically lower entropy?**
2. **Does varentropy predict correctness?**
3. **At what point in the response does the model "know" if it will be correct?**
4. **How does entropy evolve during training?**
5. **Are there specific token positions where incorrect answers show high entropy?**

These insights can help improve:
- Early stopping criteria
- Adaptive sampling strategies
- Confidence-based rejection sampling
- Understanding model failure modes

## Citation

If you use this analysis in your thesis, consider referencing:

```
Varentropy analysis inspired by:
- Entropix: https://github.com/xjdr-alt/entropix
- Research on using entropy for LLM confidence estimation
```
