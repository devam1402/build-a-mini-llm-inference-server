"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    if temperature <= 0:
        return logits

    return logits / temperature

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    if k >= logits.shape[-1]:
        return logits

    result = logits.copy()

    if logits.ndim == 1:
        threshold = np.sort(logits)[-k]
        result[logits < threshold] = -np.inf
    else:
        threshold = np.sort(logits, axis=-1)[:, -k]
        result[logits < threshold[:, None]] = -np.inf

    return result

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    # Convert logits to probabilities
    probs = stable_softmax(logits)

    # Sort probabilities from largest to smallest
    sorted_indices = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(
        probs, sorted_indices, axis=-1
    )

    # Cumulative probability
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)

    # Probability mass BEFORE the current token
    cumulative_before = cumulative_probs - sorted_probs

    # Keep the token if the mass before it is still below p
    keep_sorted = cumulative_before < p

    # Convert sorted mask back to original token order
    keep_mask = np.zeros_like(keep_sorted, dtype=bool)

    np.put_along_axis(
        keep_mask,
        sorted_indices,
        keep_sorted,
        axis=-1,
    )

    # Mask rejected logits
    return np.where(keep_mask, logits, -np.inf)

# Step 5 - sample_from_probs
import numpy as np

def sample_from_probs(probs, rng):
    return int(rng.choice(len(probs), p=probs))

# Step 6 - greedy_select
def greedy_select(logits):
    return int(np.argmax(logits))

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # Step 1: Start with special tokens
    id_to_token = list(special_tokens)

    # Step 2: Collect unique characters
    chars = set()

    for text in corpus:
        for c in text:
            chars.add(c)

    # Step 3: Sort characters
    sorted_chars = sorted(chars)

    # Step 4: Append characters, skipping existing tokens
    for c in sorted_chars:
        if c not in id_to_token:
            id_to_token.append(c)

    # Step 5: Build reverse mapping
    token_to_id = {
        token: idx
        for idx, token in enumerate(id_to_token)
    }

    # Step 6: Return both mappings
    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token
    }

