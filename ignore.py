#!/usr/bin/env python3
"""
Diagnostic script to understand SMC behavior and particle collapse.
"""
import json
import numpy as np

def analyze_smc_output(output_file="data/outputs_task2_SMC.jsonl"):
    """Analyze SMC output to diagnose weight issues."""
    
    print("=" * 80)
    print("SMC OUTPUT ANALYSIS")
    print("=" * 80)
    
    with open(output_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    for idx, result in enumerate(results[:3]):  # Analyze first 3 prompts
        print(f"\n{'='*80}")
        print(f"PROMPT {idx + 1}: {result['prefix'][:60]}...")
        print('='*80)
        
        for cont in result['continuations']:
            method = cont['method']
            samples = cont['samples']
            norm_weights = cont['normalized_weights']
            
            print(f"\nMethod: {method}")
            print(f"Number of samples: {len(samples)}")
            
            # Analyze unnormalized weights
            unnorm_weights = [s['weight'] for s in samples]
            print(f"\n--- Unnormalized Weights ---")
            print(f"  Min: {min(unnorm_weights):.6e}")
            print(f"  Max: {max(unnorm_weights):.6e}")
            print(f"  Mean: {np.mean(unnorm_weights):.6e}")
            print(f"  Std: {np.std(unnorm_weights):.6e}")
            print(f"  Range: {max(unnorm_weights) / min(unnorm_weights) if min(unnorm_weights) > 0 else float('inf'):.2e}x")
            
            # Check if all equal (within floating point tolerance)
            if max(unnorm_weights) - min(unnorm_weights) < 1e-10:
                print(f"  ⚠️  PARTICLE COLLAPSE: All weights are essentially equal!")
            elif len(set([f"{w:.6e}" for w in unnorm_weights])) < len(samples) / 2:
                print(f"  ⚠️  WARNING: Many duplicate weight values")
            else:
                print(f"  ✓ Weights have variation (unique values: {len(set([f'{w:.6e}' for w in unnorm_weights]))})")
            
            # Analyze normalized weights
            print(f"\n--- Normalized Weights ---")
            print(f"  Min: {min(norm_weights):.6f}")
            print(f"  Max: {max(norm_weights):.6f}")
            print(f"  Mean: {np.mean(norm_weights):.6f} (should be ~{1.0/len(samples):.6f})")
            print(f"  Std: {np.std(norm_weights):.6f}")
            print(f"  Sum: {sum(norm_weights):.6f} (should be 1.0)")
            
            # Check diversity
            if np.std(norm_weights) < 0.001:
                print(f"  ⚠️  CRITICAL: Essentially uniform weights (std < 0.001)")
                print(f"      This means SMC has completely collapsed to uniform sampling")
            elif np.std(norm_weights) < 0.01:
                print(f"  ⚠️  WARNING: Very low diversity (std < 0.01)")
            else:
                print(f"  ✓ Some weight diversity present")
            
            # Show samples with weights
            print(f"\n--- Sample Texts & Weights ---")
            for i, sample in enumerate(samples):
                text = sample['text'][:100].replace('\n', ' ')
                weight = sample['weight']
                norm_w = norm_weights[i]
                print(f"  [{i+1}] (unnorm={weight:.4e}, norm={norm_w:.6f})")
                print(f"      \"{text}...\"")
            
            # Check text diversity
            unique_texts = len(set(s['text'] for s in samples))
            total_chars = sum(len(s['text']) for s in samples) / len(samples)
            print(f"\n--- Text Diversity ---")
            print(f"  Unique samples: {unique_texts}/{len(samples)} ({100*unique_texts/len(samples):.1f}%)")
            print(f"  Avg length: {total_chars:.1f} characters")
            
            if unique_texts == 1:
                print(f"  ⚠️  CRITICAL: All samples are IDENTICAL!")
                print(f"      Complete particle collapse occurred.")
            elif unique_texts < len(samples) / 2:
                print(f"  ⚠️  WARNING: Low diversity (< 50% unique)")
                print(f"      Significant particle collapse occurred.")
            else:
                print(f"  ✓ Good diversity")
            
            # Analyze first/last words
            first_words = [s['text'].split()[0] if s['text'].split() else '' for s in samples]
            unique_first = len(set(first_words))
            print(f"  Unique first words: {unique_first}/{len(samples)}")
            
            if unique_first == 1:
                print(f"  → All samples start with: '{first_words[0]}'")
                print(f"  → This indicates early collapse during generation")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS & RECOMMENDATIONS")
    print('='*80)
    
    print("""
WHAT'S HAPPENING:

The issue is PARTICLE COLLAPSE - a fundamental limitation of vanilla SMC:

1. At each resampling step, particles with higher weights are duplicated
2. After a few steps, all particles converge to similar/identical sequences  
3. At the final step, they generate similar tokens → similar ΔR → equal weights
4. If they generate common tokens (low reward), weights are all small AND equal

This is EXPECTED behavior for SMC with:
- Small N (e.g., N=8 particles)
- Multiple resampling steps (T_max steps)
- No look-ahead mechanism

SOLUTIONS:

1. INCREASE β (e.g., β=10 or β=20):
   - Amplifies weight differences
   - May still collapse but at least shows variation in final step
   - Risk: numerical instability

2. INCREASE N (e.g., N=32 or N=64):
   - More particles → slower collapse
   - More expensive computationally
   - Only delays the problem

3. USE TASK 3 (TSMC) with twist functions:
   - Twist functions provide "look-ahead" to prevent collapse
   - This is the PROPER solution
   - Why Task 3 exists!

4. REDUCE resampling frequency:
   - Resample every k steps instead of every step
   - Keeps more diversity but higher variance

WHAT'S ACTUALLY CORRECT:

If normalized weights are ~1/N (uniform) and samples are diverse:
   → SMC is working but collapsed to proposal distribution
   → This is actually correct behavior for standard SMC!
   
The assignment EXPECTS you to observe this and see why TSMC (Task 3) is better!
    """)


if __name__ == "__main__":
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "data/outputs_task2_SMC.jsonl"
    
    try:
        analyze_smc_output(output_file)
    except FileNotFoundError:
        print(f"Error: File '{output_file}' not found.")
        print("Run task2.py first to generate output.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
