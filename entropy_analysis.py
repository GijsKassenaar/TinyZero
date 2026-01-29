import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the entropy file
data = torch.load('entropy_step_000001.pt')

# Check what's in it
print("Keys in file:", data.keys())
print("\nShapes:")
for key, value in data.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {value}")
# Extract the data
old_entropy = data['old_entropy']      # (1024, response_len)
old_varentropy = data['old_varentropy']  # (1024, response_len)
attention_mask = data['attention_mask']  # (1024, total_len)
rewards = data.get('rewards', None)     # (1024,) - binary reward (0 or 1) per sample (if available)

# Get response length (might be 2048 based on your config)
batch_size, response_len = old_entropy.shape
print(f"\nBatch size: {batch_size}")
print(f"Response length: {response_len}")

# Create response mask (only response tokens, not padding)
response_mask = attention_mask[:, -response_len:]  # Last response_len positions

# Count valid (non-padded) tokens per response
valid_tokens_per_response = response_mask.sum(dim=1)
print(f"\nValid tokens per response:")
print(f"  Mean: {valid_tokens_per_response.float().mean():.1f}")
print(f"  Min: {valid_tokens_per_response.min()}")
print(f"  Max: {valid_tokens_per_response.max()}")
# Mask out padding tokens for analysis
old_entropy_masked = old_entropy * response_mask
old_varentropy_masked = old_varentropy * response_mask

# Compute per-response statistics (only valid tokens)
def compute_stats(tensor, mask):
    """Compute mean over valid tokens for each response"""
    valid_sum = (tensor * mask).sum(dim=1)
    valid_count = mask.sum(dim=1)
    return valid_sum / valid_count.clamp(min=1)

entropy_per_response = compute_stats(old_entropy, response_mask)
varentropy_per_response = compute_stats(old_varentropy, response_mask)

print("\nEntropy statistics across responses:")
print(f"  Mean: {entropy_per_response.mean():.3f}")
print(f"  Std: {entropy_per_response.std():.3f}")
print(f"  Min: {entropy_per_response.min():.3f}")
print(f"  Max: {entropy_per_response.max():.3f}")

print("\nVarientropy statistics across responses:")
print(f"  Mean: {varentropy_per_response.mean():.3f}")
print(f"  Std: {varentropy_per_response.std():.3f}")
print(f"  Min: {varentropy_per_response.min():.3f}")
print(f"  Max: {varentropy_per_response.max():.3f}")

# Print reward statistics if available
if rewards is not None:
    print("\nReward statistics (binary reward per sample):")
    print(f"  Mean: {rewards.mean():.3f}")
    print(f"  Std: {rewards.std():.3f}")
    print(f"  Min: {rewards.min():.3f}")
    print(f"  Max: {rewards.max():.3f}")
    print(f"  Success rate (reward > 0): {(rewards > 0).float().mean():.2%}")
    print(f"  Success rate (reward >= 1): {(rewards >= 1).float().mean():.2%}")

# 1. Distribution of mean entropy per response
num_plots = 4 if rewards is not None else 3
plt.figure(figsize=(4 * num_plots, 4))

plt.subplot(1, num_plots, 1)
plt.hist(entropy_per_response.numpy(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Mean Entropy per Response')
plt.ylabel('Count')
plt.title('Distribution of Mean Entropy')
plt.grid(alpha=0.3)

# 2. Distribution of mean varentropy per response
plt.subplot(1, num_plots, 2)
plt.hist(varentropy_per_response.numpy(), bins=50, alpha=0.7, edgecolor='black', color='orange')
plt.xlabel('Mean Varentropy per Response')
plt.ylabel('Count')
plt.title('Distribution of Mean Varentropy')
plt.grid(alpha=0.3)

# 3. Entropy vs Varentropy scatter
plt.subplot(1, num_plots, 3)
if rewards is not None:
    # Color by reward (correct vs incorrect)
    colors = rewards.numpy()
    plt.scatter(entropy_per_response.numpy(), varentropy_per_response.numpy(), 
                c=colors, cmap='RdYlGn', alpha=0.5, s=10)
    plt.colorbar(label='Reward')
else:
    plt.scatter(entropy_per_response.numpy(), varentropy_per_response.numpy(), 
                alpha=0.5, s=10)
plt.xlabel('Mean Entropy')
plt.ylabel('Mean Varentropy')
plt.title('Entropy vs Varentropy')
plt.grid(alpha=0.3)

# 4. Reward distribution (if available)
if rewards is not None:
    plt.subplot(1, num_plots, 4)
    plt.hist(rewards.numpy(), bins=50, alpha=0.7, edgecolor='black', color='green')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('Reward Distribution')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('entropy_analysis.png', dpi=150)
print("\nSaved plot to entropy_analysis.png")

# Average entropy/varentropy across batch at each position
position_entropy = (old_entropy * response_mask).sum(dim=0) / response_mask.sum(dim=0).clamp(min=1)
position_varentropy = (old_varentropy * response_mask).sum(dim=0) / response_mask.sum(dim=0).clamp(min=1)

# Plot first 500 positions
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(position_entropy[:2048].numpy())
plt.xlabel('Token Position')
plt.ylabel('Mean Entropy')
plt.title('Entropy by Position (first 500 tokens)')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(position_varentropy[:500].numpy())
plt.xlabel('Token Position')
plt.ylabel('Mean Varentropy')
plt.title('Varentropy by Position (first 500 tokens)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('position_analysis.png', dpi=150)
print("Saved plot to position_analysis.png")

# Additional analysis: Compare entropy/varentropy for successful vs failed samples
if rewards is not None:
    # Define success as reward >= 1 (or > 0 for binary)
    success_mask = rewards >= 1
    fail_mask = rewards < 1
    
    num_success = success_mask.sum().item()
    num_fail = fail_mask.sum().item()
    
    print(f"\n\nSuccess vs Failure Analysis:")
    print(f"  Successful samples: {num_success} ({100*num_success/batch_size:.1f}%)")
    print(f"  Failed samples: {num_fail} ({100*num_fail/batch_size:.1f}%)")
    
    if num_success > 0 and num_fail > 0:
        success_entropy = entropy_per_response[success_mask]
        fail_entropy = entropy_per_response[fail_mask]
        success_varentropy = varentropy_per_response[success_mask]
        fail_varentropy = varentropy_per_response[fail_mask]
        
        print(f"\nEntropy comparison:")
        print(f"  Success: mean={success_entropy.mean():.3f}, std={success_entropy.std():.3f}")
        print(f"  Failure: mean={fail_entropy.mean():.3f}, std={fail_entropy.std():.3f}")
        
        print(f"\nVarientropy comparison:")
        print(f"  Success: mean={success_varentropy.mean():.3f}, std={success_varentropy.std():.3f}")
        print(f"  Failure: mean={fail_varentropy.mean():.3f}, std={fail_varentropy.std():.3f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist([success_entropy.numpy(), fail_entropy.numpy()], 
                 bins=30, alpha=0.7, label=['Success', 'Failure'], 
                 edgecolor='black', color=['green', 'red'])
        plt.xlabel('Mean Entropy')
        plt.ylabel('Count')
        plt.title('Entropy: Success vs Failure')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist([success_varentropy.numpy(), fail_varentropy.numpy()], 
                 bins=30, alpha=0.7, label=['Success', 'Failure'], 
                 edgecolor='black', color=['green', 'red'])
        plt.xlabel('Mean Varentropy')
        plt.ylabel('Count')
        plt.title('Varentropy: Success vs Failure')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reward_comparison.png', dpi=150)
        print("\nSaved reward comparison plot to reward_comparison.png")