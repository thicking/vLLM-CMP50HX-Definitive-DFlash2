# 2x CMP50HX profile

Status: **recovered experimental configuration; not a promoted performance preset**.

The profile [`qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env`](qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env) records the Experimental B route first observed on 2026-11-15 and workload-aware pool sizing validated on 2026-11-20:

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

After GP-21.3 workload-aware pool sizing, the startup log reported 247398 KV tokens and maximum concurrency 1.12× at the configured per-request context; the previous proportional allocation reported 252313 tokens. The scalar capacity is dynamically shared and does not fit two fully occupied 221184-token requests. Independent Mamba and draft pools have separate per-group limits, so the aggregate token count alone does not determine whether a request can be admitted.

Bounded runtime evidence recorded for this configuration includes two simultaneous short requests reaching `running=2`, a streaming/tool-call pair, prefix reuse, and a 30-minute coding-agent run with 28 completed cycles and 84 tool calls. This is acceptance evidence, not an endurance guarantee.

No standardized PP/TG throughput benchmark is available for this route. The bounded GP-21.3 acceptance recorded per-request TG only for the approximately 21K pair; the longer prompt pairs generated two output tokens each and are admission/prefill checks, not decode benchmarks. See the root `TEST-REPORT.md` for exact conditions and results. Other CMP50HX profiles, alternate GPU pairs, two full-context requests, and graph replay after every lifecycle transition are not established by this preset.

This profile does not select checkpoints, GPUs, service ports, API keys, or host paths. Provide model paths and deployment settings in a local environment file and verify them with `./launcher.sh --print-config`. Do not commit local secrets or model files.
