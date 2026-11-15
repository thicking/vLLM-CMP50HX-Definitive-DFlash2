<!-- Downstream documentation for vLLM-CMP50HX-Definitive-DFlash2; see PATCHES.md. -->
# Experimental B validation record

Record date: 2026-11-15. This report separates runtime validation of the recovered Experimental B working tree from clean-build reproducibility of the published checkout and from GPU scenarios that remain open. Experimental B was exercised on the production two-CMP50HX host; it is incorrect to characterize the validation as CPU-only.

## Runtime validation on the two-CMP50HX host

### Ten-minute streaming and concurrent scheduling — 2026-11-14

At `MAX_MODEL_LEN=147456`, the bounded stream returned HTTP 200, TTFT 1.31 s, and continued for 600.03 s after first token until the client limit; it emitted 12,513 chunks / 240,939 characters. An uneven 2,364-prompt / 17-output request returned HTTP 200 in 3.46 s while the stream was active. The scheduler log contains both request IDs in one batch with `running=2`, proving overlap for this tested pair. DFlash totals reached 13,025 drafts / 33,161 accepted. Target KV capacity logged 229,376 tokens. Two requests each filling the entire 147,456-token context were not tested and exceed this capacity; the dynamic allocator result applies to the tested uneven pair. 147456 is the highest sampled passing point for that pair, not a mathematical/global limit; the sampled 160K pair did not fit.

### Current 221184 profile, reboot recovery, prefix reuse, and tool calling — 2026-11-15

After a controlled reboot, `vllm-b.service` started and passed `/health=200`; `/v1/models` reported `swift15-qwen38-awq`, context 221184. Startup showed TP2 on GPU0/1, `MAX_NUM_SEQS=2`, DFlash2=7, target FP8 KV, draft FP16 KV, prefix caching, CUDA Graphs, `qwen3_xml`, and Custom AllReduce disabled; logged KV capacity was 252313 tokens (maximum concurrency 1.14x). This capacity does not promise two full-context requests.

- Simultaneous text streaming and `lookup_city` tool request both returned HTTP 200. The scheduler reported `running=2`; the returned tool JSON was valid and DFlash counters increased.
- Three repeated calls using the same 20,925-token prompt and named tool returned HTTP 200 in 25.793 / 4.541 / 3.343 s. Prefix-hit counter advanced 0 → 18,128 → 36,256, and draft/accepted counters rose. This exercises GPU prefix restore and repeated admission; it does not prove every partial-block/cache-shift or preemption mapping.
- An uneven workload with approximately 190,126 content-prompt tokens plus a short tool request passed; sampled KV peak was 77.27%. Exact full chat-template token count was not recorded.
- Cancellation was exercised during GP-12, with no reported stuck runtime or service failure. Owner-confirmed OpenCode and Open WebUI workflows were also exercised against B. Detailed cancellation timing, exact client versions and a complete client compatibility matrix are not present in the published artifacts.

### Coding-agent endurance — 2026-11-15

An isolated coding-agent task completed 28 cycles, 84 tool calls, 28 edits, 56 reads, and 281 messages over 1,828.9 s (~30.5 minutes). It reached `completed_target`; service PID stayed stable, health remained true, and no guard/runtime error was recorded. Across 177 samples: `MemAvailable` minimum 5.601 GiB; `SwapFree` minimum 7.523 GiB; memory PSI `avg10` maximum 0; I/O PSI `avg10` maximum 4.5; GPU0/1 reached 100% utilization with maximum temperatures 60/69 °C; GPU2 was idle; no preemptions occurred. This is bounded agent endurance, not multi-hour/multi-day reliability evidence.

At checkpoint load, workers read about 12.0/11.5 GB, cgroup memory peaked at 13.0 GiB, and a sample showed about 0.6 GiB swap per worker. Separate persistent telemetry observed paging pressure during load (minimum MemAvailable 4.39 GiB, minimum SwapFree 6.703 GiB, memory PSI full `avg10` peak 49.49%) with OOM-kill delta 0. This is not evidence that memory pressure caused the historical Nov 10 reboots.

### Throughput observations — not a standardized benchmark

A B-only snapshot measured about 41.76 tok/s for one 256-token cold-prefix output and estimated 52.85 / 49.55 tok/s per request for two simultaneous 128-token outputs. The owner currently reports average generation around 90 tok/s during real tasks and stable operation. The ~90 tok/s figure is a preliminary live-use observation without a controlled prompt, context, concurrency, or measurement window. The snapshot and owner observation are not directly comparable; no Neith comparison or general speedup is claimed.

At the final read-only check on 2026-11-15, production `vllm-b.service` was active, `/health=200`, and the served model/context were `swift15-qwen38-awq` / 221184. This documents ongoing real use, not a fresh benchmark run.

## Clean-build reproducibility from the published checkout

The runtime evidence above applies to the recovered Experimental B working tree used in production. A clean native build and install **directly from this published GitHub checkout has not been reproduced**. This is an outstanding reproducibility check, not evidence that the runtime was never GPU-tested. No wheel or container image is published. NCCL runtime identity and `nvcc` availability remain unverified.

## CPU regressions

Environment: isolated Python 3.12 CPU venv at local development host; CPU-only PyTorch 2.13.0; production-equivalent `VLLM_ENFORCE_STRICT_TOOL_CALLING=0`. The upstream global `tests/conftest.py` accelerator cleanup is excluded because CPU PyTorch does not implement that CUDA cleanup fixture.

Parser tests:

```text
CUDA_VISIBLE_DEVICES= VLLM_ENFORCE_STRICT_TOOL_CALLING=0 python -m pytest -q --confcutdir=tests/parser/engine tests/parser/engine/test_qwen3_required_tool_choice.py tests/parser/engine/test_required_tool_streaming.py
3 passed, 15 warnings in 0.57s
```

DFlash/KV tests:

```text
CUDA_VISIBLE_DEVICES= VLLM_ENFORCE_STRICT_TOOL_CALLING=0 python -m pytest -q --confcutdir=tests/v1 tests/v1/core/test_dflash_mamba_concurrency_capacity.py tests/v1/spec_decode/test_dflash_cache_shift_lifecycle_cpu.py
2 passed, 15 warnings in 7.55s
```

The exact CPU cache-shift lifecycle regression previously failed on unpatched source and passed after the narrow fix. It does not establish complete CUDA or GPU lifecycle correctness. In a separate GPU runtime scenario, repeated prompt admission exercised prefix restore and cache reuse as described above.

## Static, source and profile checks

At import time, exact tracked-diff comparison against the read-only production copy succeeded: SHA-256 `4e26a8087eb3c565e87d0ae7d74ea6056358260e1d01d30fd49c91b4066b687c`, 32 changed tracked files, 882 insertions, 100 deletions. Four additional untracked tests and the existing `profiles/2xT10/` material were copied separately and hash checked where applicable. The production snapshot manifest hashes the exact imported tree before later comment-only Apache file notices; all 42 manifest entries were verified against the snapshot commit.

```text
$ bash -n build.sh launcher.sh tools/validate_profiles.sh
$ bash tools/validate_profiles.sh
profile_validation_ok total=28
```

Python compilation of the changed scheduler/KV/speculator modules passed. `git diff --check` passed. These checks do not verify native build, GPU replay, or runtime installation from a fresh checkout.

Gitleaks scanned the working tree and full Git history using redacted JSON reports. The working-tree scan reported 12 heuristic hits; the full-history scan reported 99 hits across 24 commits. Reviewed sanitized metadata classified them as generic `key` identifiers/fields, token-count parameters, model identifiers, and documentation examples with a redacted bearer placeholder. Reported secret fields were `REDACTED`; no credential value was exposed or identified in that review. Gitleaks exits nonzero on these findings: this is a reviewed-finding result, **not a clean zero-finding scan**.

No GPU workload was started as part of extracting and publishing the repository. The runtime and GPU observations in this report are from the preceding GP-12 Experimental B validation; production was not restarted or modified for repository publication.

## Remaining regression coverage — not verified

- Clean native build/install directly from the published checkout on CMP50HX.
- GPU preemption/resume and cancellation-through-preemption interaction; cancellation itself was exercised, but latency/details were not captured in the published artifacts.
- Full GDN/Mamba slot lifecycle.
- CUDA Graph replay after scheduler-slot reuse. Graph capture/startup succeeded and graphs were active during runtime tests; replay-after-reuse remains distinct and unverified.
- Full long-duration acceptance beyond the recorded 30.5-minute agent run and 10-minute stream.
- Standardized PP/TG benchmark or performance comparison against Neith.
- Root cause of the 2026-11-10 host reboots and exact reboot-time source provenance. Reboot causation remains **UNKNOWN**; observed memory pressure is not proof.
