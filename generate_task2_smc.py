from __future__ import annotations
import os
import math
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from api import FastRewardCalculator

def load_counts_and_reward(counts_dir: str, epsilon: float = 1e-5) -> FastRewardCalculator:
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=hf_token,
        torch_dtype=torch.float16  
    ).to(device)
    
    model.eval()
    eos_id = tokenizer.eos_token_id
    return tokenizer, model, eos_id


def cal_intermediate_target_dist(reward_calc: FastRewardCalculator, tokenizer, full_ids: List[int]) -> float:
    if len(full_ids) < 3:
        return 0.0
    
    tokens = tokenizer.convert_ids_to_tokens(full_ids[-3:])
    t1_str, t2_str, t3_str = tokens[0], tokens[1], tokens[2]
    
    log_prob = reward_calc.token_lm.logp(t1_str, t2_str, t3_str)
    
    reward = -log_prob
    return reward


@torch.no_grad()
def smc_for_prompt(
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

    device = model.device
    
    prompt_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    prompt_length = prompt_ids.shape[1]
    
    particles = [prompt_ids.clone() for _ in range(N)]
    log_weights = [0.0 for _ in range(N)]
    finished = [False for _ in range(N)]
    
    for t in range(max_new_tokens):
        new_particles = []
        new_log_weights = []
        new_finished = []
        
        for i in range(N):
            if finished[i]:
                new_particles.append(particles[i])
                new_log_weights.append(log_weights[i])
                new_finished.append(True)
                continue
            
            current_ids = particles[i]
            
            attention_mask = torch.ones_like(current_ids)
            
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            logits = outputs.logits[0, -1, :]
            
            top_k_logits, top_k_indices = torch.topk(logits, k)
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]
            
            new_ids = torch.cat([current_ids, next_token.view(1, 1)], dim=1)
            
            is_finished = (next_token.item() == eos_id)
            
            full_ids_list = new_ids[0].tolist()
            delta_R_t = cal_intermediate_target_dist(reward_calc, tokenizer, full_ids_list)
            
            log_incremental_weight = beta * delta_R_t
            log_incremental_weight = max(min(log_incremental_weight, 100), -100)
            
            new_log_weight = log_weights[i] + log_incremental_weight
            
            new_particles.append(new_ids)
            new_log_weights.append(new_log_weight)
            new_finished.append(is_finished)
        
        particles = new_particles
        log_weights = new_log_weights
        finished = new_finished
        
        if all(finished):
            break
        
        if t < max_new_tokens - 1 and not all(finished):
            max_log_weight = max(log_weights)
            exp_weights = [math.exp(lw - max_log_weight) for lw in log_weights]
            total_weight = sum(exp_weights)
            
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in exp_weights]
            else:
                normalized_weights = [1.0 / N] * N
            
            resampled_indices = torch.multinomial(
                torch.tensor(normalized_weights, dtype=torch.float32),
                num_samples=N,
                replacement=True
            ).tolist()
            
            particles_new = []
            finished_new = []
            log_weights_new = []
            
            for orig_idx in resampled_indices:
                particles_new.append(particles[orig_idx].clone().detach())
                finished_new.append(finished[orig_idx])
                
                log_weights_new.append(log_weights[orig_idx])
            
            particles = particles_new
            finished = finished_new
            log_weights = log_weights_new
    
    max_log_weight = max(log_weights)
    weights = [math.exp(lw - max_log_weight) for lw in log_weights]
    
    samples = []
    for i in range(N):
        generated_ids = particles[i][0, prompt_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        samples.append({
            "text": generated_text,
            "weight": weights[i]
        })
    
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1.0 / N] * N
    
    return {
        "samples": samples,
        "normalized_weights": normalized_weights
    }