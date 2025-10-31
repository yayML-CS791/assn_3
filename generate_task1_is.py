# generate_task1_is.py
"""Sequential Importance Sampling (Algorithm 1) — generation utilities.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import math
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the fast trigram API
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

def reward_sum_pos_ids(reward_calc: FastRewardCalculator, tokenizer, ids: List[int]) -> float:
    """Compute positive reward on token ids: R_sum over token trigrams.

    Inputs:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: used only to convert ids→tokens.
        ids: full scored context (prompt+continuation) token ids.

    Output:
        R_sum (float). If len(ids) < 3, return 0.0.
    """
    if len(ids) < 3:
        return 0.0

    # Convert ids to tokenizer token strings and compute trigram log-prob sum.
    # The FastRewardCalculator exposes calculate_reward_tokens(tokens, normalize=True).
    # We want the unnormalized total reward over trigrams, so pass normalize=False.
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # calculate_reward_tokens returns 0.0 for <3 tokens, so this is safe.
    total_reward = reward_calc.calculate_reward_tokens(tokens, normalize=True)
    return float(total_reward)

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

    # Ensure there is a pad token; if missing, map pad_token to eos or add [PAD]
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    eos_id = tokenizer.eos_token_id
    return tokenizer, model, eos_id

@torch.no_grad()
def topk_decode_ids(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix: str,
    max_new: int,
    k: int,
    eos_id: int,
) -> List[int]:
    """Sample one continuation with top-k proposal. Return continuation token ids (EOS excluded).

    Inputs:
      tokenizer, model: HF components from load_model.
      prefix: full prompt string fed to the model.
      max_new: continuation token budget.
      k: top-k size.
      eos_id: stopping id. Stop early if sampled.

    Output:
      gen_ids: List[int] of sampled token ids for the continuation.
    """
    enc = tokenizer(prefix, return_tensors="pt")

    # Move inputs to the model device to avoid device mismatch errors
    device = model.device
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device) if getattr(enc, "attention_mask", None) is not None else torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        do_sample=True,
        top_k=k,
        eos_token_id=eos_id,
        pad_token_id=pad_token_id,
    )

    return output[0].tolist()

def importance_sampling_for_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    reward_calc: FastRewardCalculator,
    *,
    prefix: str,
    K: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """Run SIS for one prompt and return samples with weights.

    Inputs:
      tokenizer, model, reward_calc: initialized components.
      prefix: full prompt string given to the model (instruction + space + prefix).
      K: number of continuations to sample.
      max_new_tokens: continuation budget.
      eos_id: end-of-sequence id.
      beta: reward scale.
      k: top-k for proposal.

    Output dict:
      {
        "samples": [
          {"text": str, "weight": float},
          ...
        ],
        "normalized_weights": [float, ...]   # length K
      }
    """
    samples = []
    for _ in range(K):
        gen_ids = topk_decode_ids(
            tokenizer,
            model,
            prefix,
            max_new_tokens,
            k,
            eos_id,
        )
        full_ids = tokenizer(prefix).input_ids + gen_ids
        exponent = beta * reward_sum_pos_ids(reward_calc, tokenizer, full_ids)
        weight = math.exp(exponent)
        # print(beta, exponent, weight)
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)[len(prefix):]
        samples.append({
            "text": gen_text,
            "weight": weight
        })
    # Normalize weights
    total_weight = sum(s["weight"] for s in samples)
    normalized_weights = [s["weight"] / total_weight for s in samples] if total_weight > 0 else [0.0] * K

    return {
        "samples": samples,
        "normalized_weights": normalized_weights
    }