# Build a Mini LLM Inference Server

Construct a complete LLM inference stack from scratch: sampling, tokenization, a tiny transformer with KV caching, a paged attention allocator, continuous batching with scheduling, a streaming serving API, and a throughput/latency benchmark harness. This mirrors the architecture of modern serving systems like vLLM at a digestible scale.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** stable_softmax
- [x] **2.** apply_temperature
- [x] **3.** top_k_filter
- [x] **4.** top_p_filter
- [x] **5.** sample_from_probs
- [x] **6.** greedy_select
- [x] **7.** build_vocab

---

Built on Deep-ML.
