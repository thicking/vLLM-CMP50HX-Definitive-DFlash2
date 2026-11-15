# 2x CMP50HX profile

Status: **recovered experimental configuration; not a promoted performance preset**.

The profile [`qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env`](qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env) records the active Experimental B route observed through the production process on 2026-11-15:

| Setting | Value |
| --- | --- |
| GPUs | 2 × NVIDIA CMP 50HX 20 GB, TP=2 |
| Target | Swift Qwen3.8-27B AWQ, `compressed-tensors` |
| Draft | DFlash2 GPTQ INT8, TP=2 |
| Target / draft KV | FP8 / FP16 |
| `MAX_MODEL_LEN` / `MAX_NUM_SEQS` | 221184 / 2 |
| `GPU_UTIL` / batched tokens | 0.986 / 4096 |
| Speculative method / tokens | DFlash / 7 |
| Prefix cache / Mamba mode | enabled / `align` |
| CUDA Graphs | `FULL_AND_PIECEWISE`, capture 8 and 16; DFlash full graphs enabled |
| Custom AllReduce | disabled |

The startup log reported 252313 KV tokens and maximum concurrency 1.14× at the configured per-request context. This capacity is dynamically shared and does not fit two fully occupied 221184-token requests. Independent Mamba state pools can also prevent a second request from being admitted even when the aggregate attention KV metric appears sufficient.

Bounded runtime evidence recorded for this configuration includes two simultaneous short requests reaching `running=2`, a streaming/tool-call pair, prefix reuse, and a 30-minute coding-agent run with 28 completed cycles and 84 tool calls. This is acceptance evidence, not an endurance guarantee.

No PP/TG performance table is published for this route. Throughput values derived from Monitor telemetry require a separate metric-provenance audit; no standardized clean-checkout benchmark exists. Other CMP50HX profiles, alternate GPU pairs, full-context concurrency, and graph replay after every lifecycle transition are not established by this preset.

This profile does not select checkpoints, GPUs, service ports, API keys, or host paths. Provide model paths and deployment settings in a local environment file and verify them with `./launcher.sh --print-config`. Do not commit local secrets or model files.
