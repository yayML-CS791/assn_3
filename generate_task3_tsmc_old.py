# generate_task3_tsmc_fixed.py
"""Twisted Sequential Monte Carlo (TSMC) implementation - simplified for assignment."""
from __future__ import annotations
import os
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_task1_is import topk_decode_ids
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
    """Load and configure Hugging Face model components for Sequential Importance Sampling.
    
    Args:
        model_name: Hugging Face model repository ID (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        hf_token: Authentication token for accessing gated models
        device: Target device for model placement ("cuda:0", "cpu", etc.)
        
    Returns:
        Tuple containing:
            - tokenizer: Configured AutoTokenizer with proper padding token
            - model: AutoModelForCausalLM in evaluation mode on target device
            - eos_id: End-of-sequence token ID for generation termination
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)
    eos_id = tokenizer.eos_token_id
    return tokenizer, model, eos_id

def p_llama(tokenizer, model, ids, next_id):
    """Compute the probability of next_id given the current ids using the model."""
    input_ids = torch.tensor([ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_prob = probs[next_id].item()
    return next_token_prob


def sample_topk_next_token(tokenizer, model, prefix: str, k: int) -> int:
    """Sample next token id from model restricted to top-k highest-probability tokens.

    Returns the sampled token id (int). This uses the raw model logits (no generate)
    so we can compute proposal probabilities if needed.
    """
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
        next_logits = logits[0, -1, :]
        full_probs = torch.softmax(next_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(next_logits, k)
        topk_probs = torch.softmax(topk_vals, dim=-1)
        # Sample from the top-k distribution
        choice = torch.multinomial(topk_probs, num_samples=1).item()
        chosen_id = topk_idx[choice].item()
        chosen_prob = full_probs[chosen_id].item()
    return chosen_id, chosen_prob

def cal_intermediate_target_dist(reward_calc: FastRewardCalculator, tokenizer, full_ids: List[int], beta: int) -> float:
    """
    Args:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: for ids→tokens conversion.
        full_ids: current full context ids (prompt + generated so far).

    Returns:
        float ΔR_t ≥ 0.
    """
    # Compute delta reward: R(x_{1:t}) - R(x_{1:t-1}). Use unnormalized sum over trigrams.
    tokens = tokenizer.convert_ids_to_tokens(full_ids)
    total_reward = reward_calc.calculate_reward_tokens(tokens, normalize=False)
    if len(full_ids) > 1:
        prev_tokens = tokenizer.convert_ids_to_tokens(full_ids[:-1])
        prev_reward = reward_calc.calculate_reward_tokens(prev_tokens, normalize=False)
    else:
        prev_reward = 0.0
    delta = total_reward - prev_reward
    # The intermediate target increment is exp(beta * delta)
    return math.exp(beta * delta)

def cal_twist_function(reward_calc: FastRewardCalculator, tokenizer, seq_ids: List[int], beta: int) -> float:
    """
    Inputs:
        reward_calc: FastRewardCalculator with token_lm access.
        tokenizer: to convert ids→tokens.
        seq_ids: current full context ids (prompt + generated).

    Returns:
        Expected positive delta (float) ≥ 0. For t < 2, you may return 0.0.
        
    Note:
        you are allowed to define additional helper functions if needed in FastRewardCalculator class for calculation of expectation.
    """
    # Compute the expected positive delta using the reward calculator
    if len(seq_ids) < 2:
        return 0.0
        
    # Calculate expected reward over possible next tokens over all trigrams
    # Get data from ./tinystories_ngrams/trigram_counts.json
    expected_reward = reward_calc.calculate_expected_rewards(
        (tokenizer.convert_ids_to_tokens([seq_ids[-2]])[0],
         tokenizer.convert_ids_to_tokens([seq_ids[-1]])[0])
    )
    reward = reward_calc.calculate_reward_tokens(
        tokenizer.convert_ids_to_tokens(seq_ids), normalize=False
    )
    return math.exp(beta * (expected_reward + reward))

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

    Inputs:
      tokenizer, model: HF components from load_model.
      reward_calc: FastRewardCalculator.
      prefix: full prompt string fed to the model (instruction + space + prefix).
      N: number of particles.
      max_new_tokens: continuation budget.
      eos_id: stopping id.
      beta: reward scale.
      k: top-k for base proposal.

    Output dict (minimal for eval):
      {
        "samples": [ {"text": str, "weight": float}, ... ],   # length N
        "normalized_weights": [float, ...]                    # softmax over final log_w
      }
    """
    curr_texts = [prefix for _ in range(N)]
    finished = [False for _ in range(N)]
    weights = [0.0 for _ in range(N)]

    for t in range(max_new_tokens):
        for i in range(N):
            if finished[i]:
                continue
            # Sample next token from top-k proposal
            next_id, prop_prob = sample_topk_next_token(tokenizer, model, curr_texts[i], k)
            next_token_str = tokenizer.decode([next_id], skip_special_tokens=True)
            curr_texts[i] += next_token_str

            if next_id == eos_id:
                finished[i] = True

            full_ids = tokenizer(curr_texts[i]).input_ids
            prev_ids = full_ids[:-1]

            if t < max_new_tokens - 1:
                # Update weight
                weights[i] = cal_twist_function(reward_calc, tokenizer, full_ids, beta) / (cal_twist_function(reward_calc, tokenizer, prev_ids, beta) * p_llama(tokenizer, model, prev_ids, next_id) + 1e-12)
            else :
                weights[i] = cal_intermediate_target_dist(reward_calc, tokenizer, full_ids, beta) / (cal_twist_function(reward_calc, tokenizer, prev_ids, beta) * p_llama(tokenizer, model, prev_ids, next_id) + 1e-12)

        if t < max_new_tokens - 1 and not all(finished):
            # Resampling step
            for i in range(N):
                weights_sum = sum(weights)
                if weights_sum == 0:
                    weights = [1.0 / N for _ in range(N)]
                else:
                    weights = [w / weights_sum for w in weights]
                indices = torch.multinomial(torch.tensor(weights), num_samples=N, replacement=True).tolist()
                curr_texts = [curr_texts[idx] for idx in indices]
                finished = [finished[idx] for idx in indices]

        if all(finished):
            break

    samples = []
    normalized_weights = []
    weights_sum = sum(weights)
    if weights_sum == 0:
        normalized_weights = [1.0 / N for _ in range(N)]
    else:
        normalized_weights = [w / weights_sum for w in weights]

    for i in range(N):
        gen_text = curr_texts[i][len(prefix):]
        samples.append({
            "text": gen_text,
            "weight": weights[i]
        })

    return {
        "samples": samples,
        "normalized_weights": normalized_weights
    }