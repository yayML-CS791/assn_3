#!/usr/bin/env python3
"""
Task 0: Core Baseline Decoding Implementations

Implemented Algorithms:
    1. Greedy Decoding: Deterministic selection of maximum likelihood tokens
    2. Temperature Sampling: Stochastic sampling with temperature-scaled distributions  
    3. Top-k Sampling: Restricted stochastic sampling from k most probable tokens
"""

from typing import Tuple
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, hf_token: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load and initialize HuggingFace model and tokenizer for baseline decoding.
    
    This function handles the complete model initialization pipeline including tokenizer
    configuration, model loading, device placement, and special token identification.
    Proper setup is critical for consistent baseline performance across all decoding methods.
    
    Args:
        model_name (str): HuggingFace model repository identifier 
            Examples: "meta-llama/Llama-2-7b-hf", "gpt2", "microsoft/DialoGPT-medium"
        hf_token (str): HuggingFace authentication token for accessing gated models
            Required for models like LLaMA, GPT-4, or other restricted access models
        device (str): PyTorch device specification for model placement
            Examples: "cuda:0", "cuda:1", "cpu", "mps" (for Apple Silicon)
    
    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM, int]: Model components for generation:
            - tokenizer (AutoTokenizer): Configured tokenizer with proper padding setup
            - model (AutoModelForCausalLM): Model in evaluation mode, placed on specified device  
            - eos_id (int): End-of-sequence token ID for generation termination
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)

    # Ensure there is a pad token. Some causal LM tokenizers don't set pad_token.
    # If missing, set pad_token to eos_token (safe for open-ended generation) and
    # resize the model token embeddings if we changed the tokenizer vocabulary.
    added_tokens = 0
    if tokenizer.pad_token is None:
        # prefer eos if available, otherwise add a new [PAD] token
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_tokens = 1

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Ensure model config knows pad token id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    eos_id = tokenizer.eos_token_id
    return tokenizer, model, eos_id

@torch.no_grad()
def greedy_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int) -> str:
    """Perform greedy decoding for deterministic text generation.
            
    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        
    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    # Some tokenizers may not return an attention_mask for single inputs; fallback to ones
    attention_mask = enc.attention_mask.to(model.device) if getattr(enc, "attention_mask", None) is not None else torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        eos_token_id=eos_id,
        pad_token_id=pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prefix):]

@torch.no_grad()
def temperature_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int, tau: float) -> str:
    """Perform temperature sampling for controllable stochastic text generation.

    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        tau (float): Temperature parameter controlling randomness (must be > 0)
        
    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device) if getattr(enc, "attention_mask", None) is not None else torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        eos_token_id=eos_id,
        pad_token_id=pad_token_id,
        temperature=tau,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prefix):]

@torch.no_grad()
def topk_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int, k: int) -> str:
    """Perform top-k sampling for controlled diversity in text generation.

    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        k (int): Number of top tokens to consider (must be >= 1)
        
    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device) if getattr(enc, "attention_mask", None) is not None else torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        eos_token_id=eos_id,
        pad_token_id=pad_token_id,
        top_k=k,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prefix):]

