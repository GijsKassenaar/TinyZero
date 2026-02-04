"""
Batch Entropy Analysis: Compare entropy patterns 
between correct and incorrect answers across all rollout steps.

Usage:
    python entropy_analysis_batch.py <path_to_entropy_data_dir>
    
Example:
    python entropy_analysis_batch.py checkpoints/TinyZero/entropy_test/entropy_data/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
CORRECT_PERCENTILE = 0.8   # Upper bound percentile for correct answers (e.g., 0.8 = 80th percentile)
INCORRECT_PERCENTILE = 0.8  # Lower bound percentile for incorrect answers (e.g., 0.2 = 20th percentile)
# ============================================================================

def load_all_entropy_files(data_dir):
    """Load all entropy_step_*.pt files from directory"""
    data_dir = Path(data_dir)
    entropy_files = sorted(data_dir.glob('entropy_step_*.pt'))
    
    if not entropy_files:
        print(f"No entropy files found in {data_dir}")
        print(f"Looking for files matching: entropy_step_*.pt")
        return []
    
    print(f"Found {len(entropy_files)} entropy files")
    
    all_data = []
    for file_path in entropy_files:
        try:
            data = torch.load(file_path, weights_only=False)
            step_num = int(file_path.stem.split('_')[-1])
            data['step_num'] = step_num
            data['file_path'] = str(file_path)
            all_data.append(data)
            print(f"  Loaded {file_path.name} (step {step_num})")
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
    
    return all_data

def compute_stats_per_response(tensor, mask):
    """Compute mean over valid tokens for each response"""
    valid_sum = (tensor * mask).sum(dim=1)
    valid_count = mask.sum(dim=1)
    return valid_sum / valid_count.clamp(min=1)

def moving_average(data, window_size=10):
    """Compute moving average for smoothing"""
    if len(data) < window_size:
        return data
    
    # Convert to numpy if tensor
    if torch.is_tensor(data):
        data = data.numpy()
    
    # Compute moving average using convolution
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    
    # Fix edges (they get reduced by convolution)
    for i in range(window_size // 2):
        smoothed[i] = np.mean(data[:i+window_size//2+1])
        smoothed[-(i+1)] = np.mean(data[-(i+window_size//2+1):])
    
    return smoothed

def analyze_single_step(data):
    """Analyze entropy for a single step"""
    old_entropy = data['old_entropy']  # (batch, response_len)
    attention_mask = data['attention_mask']  # (batch, total_len)
    rewards = data.get('rewards', None)  # (batch,) - binary rewards
    
    batch_size, response_len = old_entropy.shape
    response_mask = attention_mask[:, -response_len:]
    
    # Compute per-response means
    entropy_per_response = compute_stats_per_response(old_entropy, response_mask)
    
    # Compute positional averages (averaged across batch)
    position_entropy = (old_entropy * response_mask).sum(dim=0) / response_mask.sum(dim=0).clamp(min=1)
    
    results = {
        'step': data['step_num'],
        'batch_size': batch_size,
        'response_len': response_len,
        'entropy_per_response': entropy_per_response,
        'position_entropy': position_entropy,
        'response_mask': response_mask,
        'rewards': rewards
    }
    
    # Separate by correct/incorrect if rewards available
    if rewards is not None:
        correct_mask = rewards >= 1.0
        incorrect_mask = rewards < 1.0
        
        results['correct_mask'] = correct_mask
        results['incorrect_mask'] = incorrect_mask
        results['success_rate'] = correct_mask.float().mean().item()
        
        # Compute statistics for correct samples
        if correct_mask.any():
            results['correct_entropy'] = entropy_per_response[correct_mask]
            
            # Positional analysis for correct samples
            correct_response_mask = response_mask[correct_mask]
            correct_old_entropy = old_entropy[correct_mask]
            
            results['correct_position_entropy'] = (
                (correct_old_entropy * correct_response_mask).sum(dim=0) / 
                correct_response_mask.sum(dim=0).clamp(min=1)
            )
            # Find actual max valid position (where at least one sample has data)
            results['correct_max_valid_pos'] = (correct_response_mask.sum(dim=0) > 0).sum().item()
        
        # Compute statistics for incorrect samples
        if incorrect_mask.any():
            results['incorrect_entropy'] = entropy_per_response[incorrect_mask]
            
            # Positional analysis for incorrect samples
            incorrect_response_mask = response_mask[incorrect_mask]
            incorrect_old_entropy = old_entropy[incorrect_mask]
            
            results['incorrect_position_entropy'] = (
                (incorrect_old_entropy * incorrect_response_mask).sum(dim=0) / 
                incorrect_response_mask.sum(dim=0).clamp(min=1)
            )
            # Find actual max valid position (where at least one sample has data)
            results['incorrect_max_valid_pos'] = (incorrect_response_mask.sum(dim=0) > 0).sum().item()
    
    return results

def plot_positional_grid(all_results, output_dir='./', max_position=4096, num_steps=6, all_data=None):
    """Create a grid showing positional comparison at key steps during training"""
    output_dir = Path(output_dir)
    
    results_with_rewards = [r for r in all_results if 
                           'rewards' in r and r['rewards'] is not None and
                           'correct_position_entropy' in r and 
                           'incorrect_position_entropy' in r]
    
    if not results_with_rewards:
        print("No results with rewards for grid plotting")
        return
    
    # Select evenly spaced steps
    total_steps = len(results_with_rewards)
    if total_steps < num_steps:
        step_indices = list(range(total_steps))
    else:
        step_indices = [int(i * (total_steps-1) / (num_steps-1)) for i in range(num_steps)]
    
    selected_results = [results_with_rewards[i] for i in step_indices]
    
    # Create grid: 1 row (entropy) × num_steps columns
    fig, axes = plt.subplots(1, len(selected_results), figsize=(5*len(selected_results), 5))
    
    if len(selected_results) == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    print(f"\nGenerating positional grid for steps: {[r['step'] for r in selected_results]}")
    
    for col_idx, r in enumerate(selected_results):
        step = r['step']
        success_rate = r['success_rate']
        
        correct_pos_ent = r['correct_position_entropy']
        incorrect_pos_ent = r['incorrect_position_entropy']
        
        # Use the actual valid length (minimum of both correct and incorrect)
        correct_valid = r.get('correct_max_valid_pos', len(correct_pos_ent))
        incorrect_valid = r.get('incorrect_max_valid_pos', len(incorrect_pos_ent))
        actual_len = max(correct_valid, incorrect_valid)  # Use max to show all available data
        max_pos = min(max_position, actual_len)
        positions = np.arange(max_pos)
        
        # Apply moving average for smoothing
        correct_ent_smooth = moving_average(correct_pos_ent[:max_pos], window_size=10)
        incorrect_ent_smooth = moving_average(incorrect_pos_ent[:max_pos], window_size=10)
        
        # Compute 80th percentile for correct answers and 20th for incorrect
        correct_mask = r['correct_mask']
        incorrect_mask = r['incorrect_mask']
        # Load from data
        if all_data:
            data_for_step = [d for d in all_data if d['step_num'] == r['step']]
            if data_for_step:
                old_entropy = data_for_step[0]['old_entropy']
                attention_mask = data_for_step[0]['attention_mask']
                correct_old_entropy = old_entropy[correct_mask]
                incorrect_old_entropy = old_entropy[incorrect_mask]
                # Get response mask for correct samples only
                response_len = correct_old_entropy.shape[1]
                correct_response_mask = attention_mask[correct_mask][:, -response_len:]
                incorrect_response_mask = attention_mask[incorrect_mask][:, -response_len:]
                
                # Compute percentile per position for correct, only over valid tokens
                correct_ent_percentile = torch.zeros(max_pos)
                for pos in range(max_pos):
                    # Get valid entropy values at this position (where mask = 1)
                    valid_entropies = correct_old_entropy[:, pos][correct_response_mask[:, pos] > 0]
                    if len(valid_entropies) > 0:
                        correct_ent_percentile[pos] = torch.quantile(valid_entropies, CORRECT_PERCENTILE)
                    else:
                        correct_ent_percentile[pos] = 0.0
                
                # Compute percentile per position for incorrect, only over valid tokens
                incorrect_ent_percentile = torch.zeros(max_pos)
                for pos in range(max_pos):
                    valid_entropies = incorrect_old_entropy[:, pos][incorrect_response_mask[:, pos] > 0]
                    if len(valid_entropies) > 0:
                        incorrect_ent_percentile[pos] = torch.quantile(valid_entropies, INCORRECT_PERCENTILE)
                    else:
                        incorrect_ent_percentile[pos] = 0.0
                
                correct_ent_percentile_smooth = moving_average(correct_ent_percentile, window_size=10)
                incorrect_ent_percentile_smooth = moving_average(incorrect_ent_percentile, window_size=10)
            else:
                correct_ent_percentile_smooth = None
                incorrect_ent_percentile_smooth = None
        else:
            correct_ent_percentile_smooth = None
            incorrect_ent_percentile_smooth = None
        
        # Entropy subplot
        axes[col_idx].plot(positions, correct_ent_smooth, 
                          label='Correct', color='green', linewidth=2, alpha=0.8)
        axes[col_idx].plot(positions, incorrect_ent_smooth, 
                          label='Incorrect', color='red', linewidth=2, alpha=0.8)
        if correct_ent_percentile_smooth is not None:
            axes[col_idx].plot(positions, correct_ent_percentile_smooth, 
                              label=f'{int(CORRECT_PERCENTILE*100)}th percentile', color='green', linewidth=1.5, 
                              linestyle='--', alpha=0.6)
        if incorrect_ent_percentile_smooth is not None:
            axes[col_idx].plot(positions, incorrect_ent_percentile_smooth, 
                              label=f'{int(INCORRECT_PERCENTILE*100)}th percentile', color='red', linewidth=1.5, 
                              linestyle='--', alpha=0.6)
        axes[col_idx].set_title(f'Step {step}\nSuccess: {success_rate:.1%}', 
                                fontsize=11, fontweight='bold')
        axes[col_idx].set_xlabel('Position', fontsize=10)
        axes[col_idx].set_ylabel('Entropy', fontsize=11)
        axes[col_idx].grid(alpha=0.3)
        axes[col_idx].set_ylim(bottom=0)
        axes[col_idx].set_xlim(0, max_pos)  # Set x-limit to this step's length
        
        if col_idx == 0:
            axes[col_idx].legend(fontsize=9, loc='upper left')
    
    fig.suptitle('Positional Entropy Evolution During Training (10-token moving average)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = output_dir / 'positional_evolution_grid.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: positional_evolution_grid.png")

def plot_per_step_positional(all_results, output_dir='./', max_position=4096, save_all=False, all_data=None):
    """Create individual positional plots for each step"""
    output_dir = Path(output_dir)
    step_plots_dir = output_dir / 'per_step_positional'
    step_plots_dir.mkdir(exist_ok=True)
    
    results_with_rewards = [r for r in all_results if 'rewards' in r and r['rewards'] is not None]
    
    if not results_with_rewards:
        print("No results with rewards for per-step plotting")
        return
    
    print(f"\n{'='*70}")
    print(f"Generating individual positional plots for each step...")
    print(f"{'='*70}")
    
    # Determine which steps to plot
    if save_all:
        steps_to_plot = [r['step'] for r in results_with_rewards]
        print(f"  Saving all {len(steps_to_plot)} steps...")
    else:
        # Save every 5th step to avoid too many files
        steps_to_plot = [r['step'] for i, r in enumerate(results_with_rewards) if i % 5 == 0]
        print(f"  Saving every 5th step ({len(steps_to_plot)} plots)...")
        print(f"  To save all steps, set save_all=True")
    
    for r in results_with_rewards:
        if r['step'] not in steps_to_plot:
            continue
        
        step = r['step']
        
        # Skip if no correct or incorrect data
        has_correct = 'correct_position_entropy' in r
        has_incorrect = 'incorrect_position_entropy' in r
        
        if not (has_correct and has_incorrect):
            continue
        
        correct_pos_ent = r['correct_position_entropy']
        incorrect_pos_ent = r['incorrect_position_entropy']
        
        # Use actual valid length based on masks
        correct_valid = r.get('correct_max_valid_pos', len(correct_pos_ent))
        incorrect_valid = r.get('incorrect_max_valid_pos', len(incorrect_pos_ent))
        actual_len = max(correct_valid, incorrect_valid)
        max_pos = min(max_position, actual_len)
        positions = np.arange(max_pos)
        
        # Apply moving average for smoothing
        correct_ent_smooth = moving_average(correct_pos_ent[:max_pos], window_size=10)
        incorrect_ent_smooth = moving_average(incorrect_pos_ent[:max_pos], window_size=10)
        
        # Compute 80th percentile for correct answers and 20th for incorrect
        correct_mask = r['correct_mask']
        incorrect_mask = r['incorrect_mask']
        correct_response_mask = r['response_mask'][correct_mask]
        incorrect_response_mask = r['response_mask'][incorrect_mask]
        # Get the raw entropy data (need to load from original data)
        if all_data:
            data_for_step = [d for d in all_data if d['step_num'] == r['step']]
            if data_for_step:
                old_entropy = data_for_step[0]['old_entropy']
                correct_old_entropy = old_entropy[correct_mask]
                incorrect_old_entropy = old_entropy[incorrect_mask]
                
                # Compute percentile per position for correct, only over valid tokens
                correct_ent_percentile = torch.zeros(max_pos)
                for pos in range(max_pos):
                    # Get valid entropy values at this position (where mask = 1)
                    valid_entropies = correct_old_entropy[:, pos][correct_response_mask[:, pos] > 0]
                    if len(valid_entropies) > 0:
                        correct_ent_percentile[pos] = torch.quantile(valid_entropies, CORRECT_PERCENTILE)
                    else:
                        correct_ent_percentile[pos] = 0.0
                
                # Compute percentile per position for incorrect, only over valid tokens
                incorrect_ent_percentile = torch.zeros(max_pos)
                for pos in range(max_pos):
                    valid_entropies = incorrect_old_entropy[:, pos][incorrect_response_mask[:, pos] > 0]
                    if len(valid_entropies) > 0:
                        incorrect_ent_percentile[pos] = torch.quantile(valid_entropies, INCORRECT_PERCENTILE)
                    else:
                        incorrect_ent_percentile[pos] = 0.0
                
                correct_ent_percentile_smooth = moving_average(correct_ent_percentile, window_size=10)
                incorrect_ent_percentile_smooth = moving_average(incorrect_ent_percentile, window_size=10)
            else:
                correct_ent_percentile_smooth = None
                incorrect_ent_percentile_smooth = None
        else:
            correct_ent_percentile_smooth = None
            incorrect_ent_percentile_smooth = None
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Entropy by position
        ax.plot(positions, correct_ent_smooth, 
               label=f"Correct (n={len(r['correct_entropy'])})", 
               color='green', linewidth=2, alpha=0.8)
        ax.plot(positions, incorrect_ent_smooth, 
               label=f"Incorrect (n={len(r['incorrect_entropy'])})", 
               color='red', linewidth=2, alpha=0.8)
        if correct_ent_percentile_smooth is not None:
            ax.plot(positions, correct_ent_percentile_smooth, 
                   label=f'{int(CORRECT_PERCENTILE*100)}th percentile', 
                   color='green', linewidth=1.5, linestyle='--', alpha=0.6)
        if incorrect_ent_percentile_smooth is not None:
            ax.plot(positions, incorrect_ent_percentile_smooth, 
                   label=f'{int(INCORRECT_PERCENTILE*100)}th percentile', 
                   color='red', linewidth=1.5, linestyle='--', alpha=0.6)
        ax.set_xlabel('Token Position in Response', fontsize=12)
        ax.set_ylabel('Mean Entropy', fontsize=12)
        ax.set_title(f'Step {step} - Entropy by Position: Correct vs Incorrect\n'
                    f'Success Rate: {r["success_rate"]:.1%} | 10-token moving average', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, max_pos)
        
        plt.tight_layout()
        output_file = step_plots_dir / f'positional_step_{step:06d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if len(steps_to_plot) <= 20:  # Only print progress if not too many
            print(f"  Saved: positional_step_{step:06d}.png")
    
    print(f"\n✅ Per-step positional plots saved to: {step_plots_dir}/")
    print(f"   Total files: {len(list(step_plots_dir.glob('positional_step_*.png')))}")

def plot_comparison(all_results, output_dir='./', all_data=None):
    """Create comprehensive comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check if we have reward data
    has_rewards = any('rewards' in r and r['rewards'] is not None for r in all_results)
    
    if not has_rewards:
        print("\nWarning: No reward data found in any file. Skipping correct/incorrect comparison.")
        return
    
    # Filter to only steps with rewards
    results_with_rewards = [r for r in all_results if 'rewards' in r and r['rewards'] is not None]
    
    if not results_with_rewards:
        print("No results with rewards found!")
        return
    
    print(f"\nGenerating comparison plots for {len(results_with_rewards)} steps...")
    
    # Count steps with correct/incorrect samples
    steps_with_correct = sum(1 for r in results_with_rewards if 'correct_entropy' in r and len(r['correct_entropy']) > 0)
    steps_with_incorrect = sum(1 for r in results_with_rewards if 'incorrect_entropy' in r and len(r['incorrect_entropy']) > 0)
    
    print(f"  Steps with correct samples: {steps_with_correct}/{len(results_with_rewards)}")
    print(f"  Steps with incorrect samples: {steps_with_incorrect}/{len(results_with_rewards)}")
    
    if steps_with_correct < len(results_with_rewards):
        missing_correct = len(results_with_rewards) - steps_with_correct
        print(f"  ⚠️  {missing_correct} steps have NO correct answers (all incorrect)")
    
    if steps_with_incorrect < len(results_with_rewards):
        missing_incorrect = len(results_with_rewards) - steps_with_incorrect
        print(f"  ⚠️  {missing_incorrect} steps have NO incorrect answers (all correct)")
    
    # ========== PLOT 1: Success Rate Over Time ==========
    plt.figure(figsize=(10, 5))
    steps = [r['step'] for r in results_with_rewards]
    success_rates = [r['success_rate'] for r in results_with_rewards]
    
    plt.plot(steps, success_rates, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate Across Training Steps', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_over_time.png', dpi=150)
    print(f"  Saved: success_rate_over_time.png")
    plt.close()
    
    # ========== PLOT 2: Mean Entropy: Correct vs Incorrect ==========
    plt.figure(figsize=(10, 5))
    
    # Extract statistics with corresponding steps
    correct_entropy_means = []
    correct_entropy_steps = []
    incorrect_entropy_means = []
    incorrect_entropy_steps = []
    
    for r in results_with_rewards:
        if 'correct_entropy' in r and len(r['correct_entropy']) > 0:
            correct_entropy_means.append(r['correct_entropy'].mean().item())
            correct_entropy_steps.append(r['step'])
        if 'incorrect_entropy' in r and len(r['incorrect_entropy']) > 0:
            incorrect_entropy_means.append(r['incorrect_entropy'].mean().item())
            incorrect_entropy_steps.append(r['step'])
    
    # Entropy comparison
    if correct_entropy_means:
        plt.plot(correct_entropy_steps, correct_entropy_means, marker='o', label='Correct', 
                color='green', linewidth=2, markersize=6)
    if incorrect_entropy_means:
        plt.plot(incorrect_entropy_steps, incorrect_entropy_means, marker='s', label='Incorrect', 
                color='red', linewidth=2, markersize=6)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Mean Entropy', fontsize=12)
    plt.title('Entropy: Correct vs Incorrect', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_comparison.png', dpi=150)
    print(f"  Saved: entropy_comparison.png")
    plt.close()
    
    # ========== PLOT 3: Positional Analysis (Average Across All Steps) ==========
    # Aggregate position data across all steps
    all_correct_position_entropy = []
    all_incorrect_position_entropy = []
    
    for r in results_with_rewards:
        if 'correct_position_entropy' in r:
            all_correct_position_entropy.append(r['correct_position_entropy'])
        if 'incorrect_position_entropy' in r:
            all_incorrect_position_entropy.append(r['incorrect_position_entropy'])
    
    if all_correct_position_entropy and all_incorrect_position_entropy:
        # Stack and average
        correct_pos_ent = torch.stack(all_correct_position_entropy).mean(dim=0)
        incorrect_pos_ent = torch.stack(all_incorrect_position_entropy).mean(dim=0)
        
        # Use actual max length across all steps
        actual_max_len = max(len(pos_ent) for pos_ent in all_correct_position_entropy)
        max_pos = min(4096, actual_max_len)
        positions = np.arange(max_pos)
        
        print(f"  Positional plot using {max_pos} tokens (actual max length across steps)")
        
        # Apply moving average for smoothing
        correct_ent_smooth = moving_average(correct_pos_ent[:max_pos], window_size=10)
        incorrect_ent_smooth = moving_average(incorrect_pos_ent[:max_pos], window_size=10)
        correct_varent_smooth = moving_average(correct_pos_varent[:max_pos], window_size=10)
        incorrect_varent_smooth = moving_average(incorrect_pos_varent[:max_pos], window_size=10)
        
        # Compute 80th percentile across all steps for correct answers and 20th for incorrect
        all_correct_entropy_raw = []
        all_correct_masks_raw = []
        all_incorrect_entropy_raw = []
        all_incorrect_masks_raw = []
        if all_data:
            for r in results_with_rewards:
                if 'correct_mask' in r and 'incorrect_mask' in r:
                    data_for_step = [d for d in all_data if d['step_num'] == r['step']]
                    if data_for_step:
                        old_entropy = data_for_step[0]['old_entropy']
                        attention_mask = data_for_step[0]['attention_mask']
                        correct_mask = r['correct_mask']
                        incorrect_mask = r['incorrect_mask']
                        correct_old_entropy = old_entropy[correct_mask]
                        incorrect_old_entropy = old_entropy[incorrect_mask]
                        # Get response mask for correct and incorrect samples
                        response_len = correct_old_entropy.shape[1]
                        correct_response_mask = attention_mask[correct_mask][:, -response_len:]
                        incorrect_response_mask = attention_mask[incorrect_mask][:, -response_len:]
                        all_correct_entropy_raw.append(correct_old_entropy)
                        all_correct_masks_raw.append(correct_response_mask)
                        all_incorrect_entropy_raw.append(incorrect_old_entropy)
                        all_incorrect_masks_raw.append(incorrect_response_mask)
        
        correct_ent_percentile_smooth = None
        incorrect_ent_percentile_smooth = None
        
        if all_correct_entropy_raw:
            # Concatenate all correct samples across steps
            all_correct_concat = torch.cat(all_correct_entropy_raw, dim=0)
            all_correct_masks_concat = torch.cat(all_correct_masks_raw, dim=0)
            
            # Compute percentile per position, only over valid tokens
            correct_ent_percentile = torch.zeros(max_pos)
            for pos in range(max_pos):
                # Get valid entropy values at this position (where mask = 1)
                valid_entropies = all_correct_concat[:, pos][all_correct_masks_concat[:, pos] > 0]
                if len(valid_entropies) > 0:
                    correct_ent_percentile[pos] = torch.quantile(valid_entropies, CORRECT_PERCENTILE)
                else:
                    correct_ent_percentile[pos] = 0.0
            
            correct_ent_percentile_smooth = moving_average(correct_ent_percentile, window_size=10)
        
        if all_incorrect_entropy_raw:
            # Concatenate all incorrect samples across steps
            all_incorrect_concat = torch.cat(all_incorrect_entropy_raw, dim=0)
            all_incorrect_masks_concat = torch.cat(all_incorrect_masks_raw, dim=0)
            
            # Compute percentile per position, only over valid tokens
            incorrect_ent_percentile = torch.zeros(max_pos)
            for pos in range(max_pos):
                # Get valid entropy values at this position (where mask = 1)
                valid_entropies = all_incorrect_concat[:, pos][all_incorrect_masks_concat[:, pos] > 0]
                if len(valid_entropies) > 0:
                    incorrect_ent_percentile[pos] = torch.quantile(valid_entropies, INCORRECT_PERCENTILE)
                else:
                    incorrect_ent_percentile[pos] = 0.0
            
            incorrect_ent_percentile_smooth = moving_average(incorrect_ent_percentile, window_size=10)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Entropy by position
        ax.plot(positions, correct_ent_smooth, 
               label='Correct', color='green', linewidth=2, alpha=0.8)
        ax.plot(positions, incorrect_ent_smooth, 
               label='Incorrect', color='red', linewidth=2, alpha=0.8)
        if correct_ent_percentile_smooth is not None:
            ax.plot(positions, correct_ent_percentile_smooth, 
                   label=f'{int(CORRECT_PERCENTILE*100)}th percentile', 
                   color='green', linewidth=1.5, linestyle='--', alpha=0.6)
        if incorrect_ent_percentile_smooth is not None:
            ax.plot(positions, incorrect_ent_percentile_smooth, 
                   label=f'{int(INCORRECT_PERCENTILE*100)}th percentile', 
                   color='red', linewidth=1.5, linestyle='--', alpha=0.6)
        ax.set_xlabel('Token Position in Response', fontsize=12)
        ax.set_ylabel('Mean Entropy', fontsize=12)
        ax.set_title('Entropy by Position: Correct vs Incorrect (Averaged Across Steps)\n10-token moving average', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'positional_comparison.png', dpi=150)
        print(f"  Saved: positional_comparison.png")
        plt.close()
    
    # ========== PLOT 4: Entropy Distribution (All Steps Combined) ==========
    all_correct_entropy = torch.cat([r['correct_entropy'] for r in results_with_rewards 
                                      if 'correct_entropy' in r])
    all_incorrect_entropy = torch.cat([r['incorrect_entropy'] for r in results_with_rewards 
                                        if 'incorrect_entropy' in r])
    
    plt.figure(figsize=(10, 6))
    
    # Entropy distributions
    plt.hist([all_correct_entropy.numpy(), all_incorrect_entropy.numpy()],
            bins=50, alpha=0.7, label=['Correct', 'Incorrect'],
            color=['green', 'red'], edgecolor='black')
    plt.xlabel('Mean Entropy per Response', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Entropy Distribution (All Steps)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_distribution.png', dpi=150)
    print(f"  Saved: entropy_distribution.png")
    plt.close()
    
    print("\n✅ All comparison plots generated successfully!")

def print_summary_statistics(all_results):
    """Print summary statistics across all steps"""
    results_with_rewards = [r for r in all_results if 'rewards' in r and r['rewards'] is not None]
    
    if not results_with_rewards:
        print("\nNo reward data available for summary statistics.")
        return
    
    print("\n" + "="*70)
    print(" SUMMARY STATISTICS ACROSS ALL STEPS")
    print("="*70)
    
    # Overall statistics
    total_samples = sum(r['batch_size'] for r in results_with_rewards)
    total_correct = sum((r['rewards'] >= 1.0).sum().item() for r in results_with_rewards)
    overall_success_rate = total_correct / total_samples
    
    print(f"\nOverall:")
    print(f"  Total steps analyzed: {len(results_with_rewards)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Total correct: {total_correct}")
    print(f"  Overall success rate: {overall_success_rate:.2%}")
    
    # Aggregate all data
    all_correct_entropy = torch.cat([r['correct_entropy'] for r in results_with_rewards 
                                      if 'correct_entropy' in r and len(r['correct_entropy']) > 0])
    all_incorrect_entropy = torch.cat([r['incorrect_entropy'] for r in results_with_rewards 
                                        if 'incorrect_entropy' in r and len(r['incorrect_entropy']) > 0])
    
    print(f"\nEntropy Statistics:")
    print(f"  Correct answers:")
    print(f"    Mean: {all_correct_entropy.mean():.4f} ± {all_correct_entropy.std():.4f}")
    print(f"    Median: {all_correct_entropy.median():.4f}")
    print(f"    Range: [{all_correct_entropy.min():.4f}, {all_correct_entropy.max():.4f}]")
    print(f"  Incorrect answers:")
    print(f"    Mean: {all_incorrect_entropy.mean():.4f} ± {all_incorrect_entropy.std():.4f}")
    print(f"    Median: {all_incorrect_entropy.median():.4f}")
    print(f"    Range: [{all_incorrect_entropy.min():.4f}, {all_incorrect_entropy.max():.4f}]")
    print(f"  Difference (Incorrect - Correct): {all_incorrect_entropy.mean() - all_correct_entropy.mean():.4f}"
    
    print("\n" + "="*70)

def analyze_group_patterns(all_data, group_size=4):
    """Analyze patterns in groups of responses (using UIDs if available, otherwise assumes sequential groups)"""
    print("\n" + "="*70)
    print(f" GROUP PATTERN ANALYSIS")
    print("="*70)
    
    results_with_rewards = [data for data in all_data if 'rewards' in data and data['rewards'] is not None]
    
    if not results_with_rewards:
        print("No reward data available for group analysis.")
        return None
    
    # Check if UIDs are available (check all files, not just first)
    has_uids = all('uids' in data for data in results_with_rewards)
    if has_uids:
        print(f"✓ UIDs found - using actual GRPO groupings")
    else:
        # Check which files have UIDs for debugging
        files_with_uids = sum(1 for data in results_with_rewards if 'uids' in data)
        print(f"⚠ No UIDs in all files ({files_with_uids}/{len(results_with_rewards)} have UIDs)")
        print(f"  Assuming sequential groups of {group_size}")
        print(f"  (Note: Groups may be inaccurate due to load balancing)")
    
    all_group_stats = []
    
    for data in results_with_rewards:
        step_num = data['step_num']
        rewards = data['rewards']
        batch_size = len(rewards)
        
        group_patterns = defaultdict(int)
        
        # Check if this specific file has UIDs
        if 'uids' in data:
            # Use UIDs to identify actual GRPO groups
            uids = data['uids']
            uid_to_rewards = defaultdict(list)
            
            # Group rewards by UID
            for i, uid in enumerate(uids):
                uid_to_rewards[uid].append(rewards[i])
            
            # Analyze each group
            num_groups = len(uid_to_rewards)
            for uid, group_rewards in uid_to_rewards.items():
                group_rewards_tensor = torch.tensor(group_rewards) if not isinstance(group_rewards[0], torch.Tensor) else torch.stack(group_rewards)
                num_correct = (group_rewards_tensor >= 1.0).sum().item()
                group_size_actual = len(group_rewards)
                group_patterns[num_correct] += 1
                
                # Warn if group size doesn't match expected
                if group_size_actual != group_size:
                    print(f"  Warning: Group {uid[:8]}... has {group_size_actual} samples (expected {group_size})")
        else:
            # Fallback: assume sequential groups (may be inaccurate after load balancing!)
            if batch_size % group_size != 0:
                print(f"\nWarning: Step {step_num} batch size {batch_size} not divisible by {group_size}")
                continue
            
            num_groups = batch_size // group_size
            for i in range(num_groups):
                group_rewards = rewards[i*group_size:(i+1)*group_size]
                num_correct = (group_rewards >= 1.0).sum().item()
                group_patterns[num_correct] += 1
        
        all_group_stats.append({
            'step': step_num,
            'num_groups': num_groups,
            'patterns': dict(group_patterns),
            'used_uids': has_uids
        })
    
    # Print summary
    print(f"\nAnalyzed {len(all_group_stats)} steps")
    print(f"Total groups per step: {all_group_stats[0]['num_groups'] if all_group_stats else 0}")
    print(f"Grouping method: {'UID-based (accurate)' if has_uids else f'Sequential assumption (may be inaccurate)'}")
    
    # Aggregate patterns across all steps
    total_patterns = defaultdict(int)
    for stat in all_group_stats:
        for num_correct, count in stat['patterns'].items():
            total_patterns[num_correct] += count
    
    total_groups = sum(total_patterns.values())
    
    print(f"\nOverall Group Composition (across all steps):")
    for num_correct in sorted(total_patterns.keys()):
        count = total_patterns[num_correct]
        percentage = 100 * count / total_groups
        print(f"  {num_correct}/{group_size} correct: {count:5d} groups ({percentage:5.1f}%)")
    
    # Show evolution over training
    print(f"\nGroup Pattern Evolution:")
    print(f"{'Step':<10} {'0/4':>8} {'1/4':>8} {'2/4':>8} {'3/4':>8} {'4/4':>8}")
    print("-" * 54)
    
    for stat in all_group_stats:
        step = stat['step']
        patterns = stat['patterns']
        num_groups = stat['num_groups']
        
        row = f"{step:<10}"
        for i in range(group_size + 1):
            count = patterns.get(i, 0)
            pct = 100 * count / num_groups if num_groups > 0 else 0
            row += f" {pct:6.1f}%"
        print(row)
    
    print("\n" + "="*70)
    
    # Create stacked bar plot showing group composition evolution
    if all_group_stats:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        steps = [stat['step'] for stat in all_group_stats]
        
        # Prepare data for stacked bar chart
        percentages = {i: [] for i in range(group_size + 1)}
        
        for stat in all_group_stats:
            patterns = stat['patterns']
            num_groups = stat['num_groups']
            
            for i in range(group_size + 1):
                count = patterns.get(i, 0)
                pct = 100 * count / num_groups if num_groups > 0 else 0
                percentages[i].append(pct)
        
        # Create stacked bar chart
        colors = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c']  # Red to green gradient
        bottom = np.zeros(len(steps))
        
        for i in range(group_size + 1):
            ax.bar(steps, percentages[i], bottom=bottom, label=f'{i}/{group_size} correct',
                   color=colors[i], edgecolor='white', linewidth=0.5)
            bottom += np.array(percentages[i])
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Percentage of Groups (%)', fontsize=12)
        ax.set_title(f'GRPO Group Composition Evolution During Training\n'
                     f'(Grouping: {"UID-based" if has_uids else "Sequential assumption"})',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save to output directory (will be determined by caller)
        output_file = Path(all_data[0]['file_path']).parent.parent / 'entropy_analysis' / 'group_composition_evolution.png'
        output_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved group composition plot to: {output_file.name}")
    
    return all_group_stats

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide path to entropy data directory")
        print("\nExample usage:")
        print("  python entropy_analysis_batch.py checkpoints/TinyZero/entropy_test/entropy_data/")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    print(f"Loading entropy data from: {data_dir}")
    print("="*70)
    
    # Load all files
    all_data = load_all_entropy_files(data_dir)
    
    if not all_data:
        print("\nNo data to analyze!")
        sys.exit(1)
    
    # Analyze each step
    print(f"\n{'='*70}")
    print("Analyzing entropy for each step...")
    print("="*70)
    
    all_results = []
    for data in all_data:
        results = analyze_single_step(data)
        all_results.append(results)
        
        print(f"\nStep {results['step']}:")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Response length: {results['response_len']}")
        
        if results.get('rewards') is not None:
            print(f"  Success rate: {results['success_rate']:.2%}")
            if 'correct_entropy' in results and len(results['correct_entropy']) > 0:
                print(f"  Correct - Entropy: {results['correct_entropy'].mean():.4f} ± {results['correct_entropy'].std():.4f}")
            if 'incorrect_entropy' in results and len(results['incorrect_entropy']) > 0:
                print(f"  Incorrect - Entropy: {results['incorrect_entropy'].mean():.4f} ± {results['incorrect_entropy'].std():.4f}")
    
    # Print summary statistics
    print_summary_statistics(all_results)
    
    # Analyze group patterns (groups of 4 responses to same prompt)
    analyze_group_patterns(all_data, group_size=4)
    
    # Generate comparison plots
    print(f"\n{'='*70}")
    print("Generating comparison plots...")
    print("="*70)
    output_dir = Path(data_dir).parent / 'entropy_analysis'
    output_dir.mkdir(exist_ok=True)
    plot_comparison(all_results, output_dir, all_data=all_data)
    
    # Generate positional evolution grid (overview)
    plot_positional_grid(all_results, output_dir, max_position=4096, num_steps=6, all_data=all_data)
    
    # Generate per-step positional plots (detailed)
    plot_per_step_positional(all_results, output_dir, max_position=4096, save_all=False, all_data=all_data)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")
    print(f"\nGenerated plots:")
    print(f"  1. success_rate_over_time.png")
    print(f"  2. entropy_comparison.png")
    print(f"  3. positional_comparison.png (averaged across all steps)")
    print(f"  4. entropy_distribution.png")
    print(f"  5. positional_evolution_grid.png (6 key steps side-by-side)")
    print(f"  6. per_step_positional/ (individual detailed plots)")
    print(f"\nTo save all step plots (not just every 5th), edit the call: save_all=True")

if __name__ == '__main__':
    main()
