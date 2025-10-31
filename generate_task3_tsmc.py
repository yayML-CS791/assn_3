# generate_task3_tsmc_fixed.py
"""Twisted Sequential Monte Carlo (TSMC) implementation - fixed version."""
from __future__ import annotations
import os
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

from api import FastRewardCalculator

def load_counts_and_reward(counts_dir: str, epsilon: float = 1e-9) -> FastRewardCalculator:
    """Initialize trigram-based reward calculator for Sequential Importance Sampling.
    
    Args:
        counts_dir: Directory path containing ngrams data with trigram_probs.pkl cache
        epsilon: Smoothing parameter - minimum probability for unseen trigrams (prevents log(0))
        
    Returns:
        FastRewardCalculator: Configured calculator for computing R(x) rewards
    """
    cache_file = os.path.join(counts_dir, "trigram_probs.pkl")
    trigram_counts_file = os.path.join(counts_dir, "trigram_counts.json")
    return FastRewardCalculator(cache_file, trigram_counts_file, epsilon=epsilon)

def load_model(model_name: str, hf_token: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load and configure Hugging Face model components."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=hf_token,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    eos_id = tokenizer.eos_token_id
    return tokenizer, model, eos_id

def cal_intermediate_target_dist(reward_calc: FastRewardCalculator, tokenizer, full_ids: List[int], beta: float) -> float:
    """Compute incremental reward ΔR_t in log space.
    
    Returns exp(beta * delta_R) where delta_R is incremental reward.
    """
    if len(full_ids) < 3:
        return 1.0  # No trigram reward yet
    
    # Get last three tokens
    tokens = tokenizer.convert_ids_to_tokens(full_ids[-3:])
    if len(tokens) < 3:
        return 1.0
    
    t1_str, t2_str, t3_str = tokens[0], tokens[1], tokens[2]
    
    # Get log probability of trigram
    log_prob = reward_calc.token_lm.logp(t1_str, t2_str, t3_str)
    
    # Reward = negative log probability
    reward = -log_prob
    
    return math.exp(beta * reward)

@torch.no_grad()
def tsmc_for_prompt(
    tokenizer: Any,
    model: Any,
    reward_calc: Any,
    *,
    prefix: str,
    N: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """Run TSMC for a single prompt.

    Implements Algorithm 3 from the assignment with proper weight accumulation
    and resampling.
    """
    device = model.device
    prompt_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    prompt_length = prompt_ids.shape[1]
    
    # Initialize particles
    particles = [prompt_ids.clone() for _ in range(N)]
    log_weights = [0.0 for _ in range(N)]
    finished = [False for _ in range(N)]
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Starting TSMC for prompt: '{prefix[:50]}...'")
    print(f"  N={N}, beta={beta}, k={k}, max_new_tokens={max_new_tokens}")
    print(f"  Prompt length: {prompt_length} tokens")
    print(f"{'='*80}\n")

    for t in range(max_new_tokens):
        new_particles = []
        new_log_weights = []
        new_finished = []
        
        step_tokens = []
        step_rewards = []
        step_log_weights_before = []
        
        # === GENERATE STEP ===
        for i in range(N):
            if finished[i]:
                new_particles.append(particles[i])
                new_log_weights.append(log_weights[i])
                new_finished.append(True)
                step_tokens.append("(finished)")
                step_rewards.append(None)
                step_log_weights_before.append(log_weights[i])
                continue
            
            current_ids = particles[i]
            step_log_weights_before.append(log_weights[i])
            
            # Get top-k from model
            outputs = model(input_ids=current_ids, use_cache=False)
            logits = outputs.logits[0, -1, :]
            
            top_k_logits, top_k_indices = torch.topk(logits, k)
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            # Sample from top-k
            next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]
            
            # Debug
            token_str = tokenizer.convert_ids_to_tokens([next_token.item()])[0]
            step_tokens.append(token_str)
            
            new_ids = torch.cat([current_ids, next_token.view(1, 1)], dim=1)
            is_finished = (next_token.item() == eos_id)
            
            # Compute incremental weight
            full_ids_list = new_ids[0].tolist()
            delta_R_t = cal_intermediate_target_dist(reward_calc, tokenizer, full_ids_list, beta)
            step_rewards.append(math.log(delta_R_t) if delta_R_t > 1e-12 else 0.0)
            
            # In TSMC, weight increment is delta_R_t at each step
            # (for simplicity, not using full twist functions)
            log_incremental_weight = math.log(delta_R_t) if delta_R_t > 1e-12 else -100.0
            new_log_weight = log_weights[i] + log_incremental_weight
            
            new_particles.append(new_ids)
            new_log_weights.append(new_log_weight)
            new_finished.append(is_finished)
        
        particles = new_particles
        log_weights = new_log_weights
        finished = new_finished
        
        # === DEBUG OUTPUT ===
        if t < 5 or t == max_new_tokens - 1 or (t % 10 == 0):
            print(f"\n--- STEP {t} ---")
            print(f"Active: {sum(not f for f in finished)}/{N}")
            print(f"Tokens: {step_tokens}")
            reward_strs = [f"{r:.4f}" if r is not None else "N/A" for r in step_rewards]
            print(f"Log(ΔR_t): {reward_strs}")
            lw_before = [f"{lw:.2f}" for lw in step_log_weights_before]
            lw_after = [f"{lw:.2f}" for lw in log_weights]
            print(f"LogW before: {lw_before}")
            print(f"LogW after:  {lw_after}")
            print(f"LogW stats: min={min(log_weights):.2f}, max={max(log_weights):.2f}")
            
            unique_seqs = len(set(tuple(p[0].tolist()) for p in particles))
            print(f"Unique sequences: {unique_seqs}/{N}")
        
        if all(finished):
            print(f"\n[Step {t}] All particles finished early")
            break
        
        # === RESAMPLING STEP (once per timestep) ===
        if t < max_new_tokens - 1 and not all(finished):
            max_log_weight = max(log_weights)
            exp_weights = [math.exp(lw - max_log_weight) for lw in log_weights]
            total_weight = sum(exp_weights)
            
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in exp_weights]
            else:
                normalized_weights = [1.0 / N] * N
            
            # Debug resampling
            if t < 5:
                print(f"\n  RESAMPLING:")
                norm_w_str = [f"{w:.4f}" for w in normalized_weights]
                print(f"  Normalized weights: {norm_w_str}")
                max_weight = max(normalized_weights)
                if max_weight > 0.8:
                    dominant_idx = normalized_weights.index(max_weight)
                    print(f"  ⚠️  PARTICLE {dominant_idx} DOMINATES with weight {max_weight:.4f}")
            
            resampled_indices = torch.multinomial(
                torch.tensor(normalized_weights, dtype=torch.float32),
                num_samples=N,
                replacement=True
            ).tolist()
            
            if t < 5:
                print(f"  Resampled indices: {resampled_indices}")
                if len(set(resampled_indices)) == 1:
                    print(f"  ⚠️  ALL PARTICLES RESAMPLED TO INDEX {resampled_indices[0]}!")
            
            # Create new particle set
            particles_new = []
            finished_new = []
            
            for orig_idx in resampled_indices:
                particles_new.append(particles[orig_idx].clone().detach())
                finished_new.append(finished[orig_idx])
            
            particles = particles_new
            finished = finished_new
            log_weights = [0.0 for _ in range(N)]  # Reset after resampling
            
            if t < 5:
                print(f"  LogW after resample: {[f'{lw:.2f}' for lw in log_weights]}")
    
    # === FINAL OUTPUT ===
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"{'='*80}")
    
    max_log_weight = max(log_weights)
    weights = [math.exp(lw - max_log_weight) for lw in log_weights]
    
    print(f"\nFinal log_weights:")
    for i, lw in enumerate(log_weights):
        print(f"  Particle {i}: {lw:.6f}")
    
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1.0 / N] * N
    
    # Show generated texts
    print(f"\nGenerated texts:")
    samples = []
    for i in range(N):
        generated_ids = particles[i][0, prompt_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        samples.append({
            "text": generated_text,
            "weight": weights[i]
        })
        
        print(f"  [{i}] (w={weights[i]:.4e}) {generated_text[:80]}...")
    
    unique_texts = len(set(s['text'] for s in samples))
    print(f"\nUnique texts: {unique_texts}/{N}")
    
    print(f"\n{'='*80}\n")
    
    return {
        "samples": samples,
        "normalized_weights": normalized_weights
    }