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

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # Step 1: Get the token-to-ID dictionary
    token_to_id = vocab["token_to_id"]

    # Step 2: Start with an empty list
    ids = []

    # Step 3: Add BOS if requested and available
    if add_bos and "<bos>" in token_to_id:
        ids.append(token_to_id["<bos>"])

    # Step 4: Encode every character
    for c in text:

        # Case A: Character exists in vocabulary
        if c in token_to_id:
            ids.append(token_to_id[c])

        # Case B: Character is unknown, use UNK if available
        elif "<unk>" in token_to_id:
            ids.append(token_to_id["<unk>"])

        # Case C: No UNK token, skip character
        else:
            continue

    # Step 5: Return the flat list of token IDs
    return ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):

    # Step 1: Get the ID-to-token mapping
    id_to_token = vocab["id_to_token"]

    # Step 2: Store decoded token strings
    pieces = []

    # Step 3: Process each token ID
    for tid in token_ids:

        # Step 4: Convert NumPy integer to Python integer
        tid = int(tid)

        # Step 5: Look up the corresponding token
        token = id_to_token[tid]

        # Step 6: Identify and skip special tokens
        is_special = (
            len(token) >= 2
            and token.startswith("<")
            and token.endswith(">")
        )

        if skip_special and is_special:
            continue

        # Step 7: Keep ordinary tokens
        pieces.append(token)

    # Step 8: Combine tokens into one string
    return "".join(pieces)

# Step 14 - causal_attention
import numpy as np


def causal_attention(q, k, v, is_causal=True):
    """
    Compute scaled dot-product attention.

    Args:
        q: Query matrix, shape (Tq, D)
        k: Key matrix, shape (Tk, D)
        v: Value matrix, shape (Tk, Dv)
        is_causal: Whether to apply a causal attention mask.

    Returns:
        Output matrix, shape (Tq, Dv)
    """

    # 1. Calculate the query-key similarity scores.
    scores = q @ k.T

    # 2. Scale scores by the square root of the head dimension.
    d = q.shape[-1]
    scores = scores / np.sqrt(d)

    # 3. Apply the causal mask when requested.
    if is_causal:
        tq = q.shape[0]
        tk = k.shape[0]

        # For cached decoding, query positions correspond
        # to the final Tq positions in the key sequence.
        query_positions = np.arange(tq) + (tk - tq)
        key_positions = np.arange(tk)

        # A query cannot attend to keys from future positions.
        mask = key_positions[None, :] > query_positions[:, None]

        scores = np.where(mask, -np.inf, scores)

    # 4. Numerically stable softmax.
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    # 5. Weighted sum of the value vectors.
    output = attention_weights @ v

    return output

