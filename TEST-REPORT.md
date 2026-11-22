<!-- Downstream documentation for vLLM-CMP50HX-Definitive-DFlash2; see PATCHES.md. -->
# Experimental B validation record

## v0.2.0-alpha.2 documentation supplement — 2026-11-22

This documentation-only release adds [lessons learned](docs/lessons-learned.ru.md), a [Prefix Cache guide](docs/prefix-cache.md), and [profile-specific measurements](docs/benchmarks.md). It does not change runtime code or configuration and does not add GPU tests. The recorded values are single observed runs where stated; they are not generalized performance guarantees. Existing runtime validation remains distinct from clean-build reproducibility; a clean build directly from the published checkout remains **NOT VERIFIED**.

## v0.2.0-alpha.1 release verification — 2026-11-20

### Targeted CPU regressions

The first harness attempt failed before six tests reached their assertions because CPU platform selection was left at `auto` and this host has no accelerator. No implementation defect was established by that attempt. After setting `VLLM_TARGET_DEVICE=cpu`, the same focused set passed:

```text
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= VLLM_ENFORCE_STRICT_TOOL_CALLING=0 \
  python -m pytest -q --confcutdir=tests/v1 \
  tests/v1/metrics/test_stats.py tests/v1/core/test_scheduler.py \
  tests/v1/core/test_kv_cache_utils.py tests/v1/engine/test_engine_args.py \
  -k 'live_prefill_prometheus or live_prefill_metrics or prefill_barrier_does_not_promote_waiters_past_blocked_fifo_head or dflash_workload_pool_sizing or dflash_workload_sizing_rejects_explicit_allocator_override or dflash_draft_pool_concurrency_from_cli'
9 passed, 305 deselected, 15 warnings in 25.44s
```

The selected tests cover live prefill telemetry without output tokens, mixed prefill/decode and cancellation accounting, the multi-waiter barrier livelock reproducer, workload-aware allocator lifecycle/fail-closed behavior, and CLI parsing of the new setting. This was a targeted CPU run, not the full upstream suite and not GPU acceptance.

Additional release checks:

```text
bash -n build.sh launcher.sh tools/validate_profiles.sh tests/standalone_tests/launcher_profile_schema.sh tests/standalone_tests/profile_validator.sh
bash tools/validate_profiles.sh
profile_validation_ok total=28
bash tests/standalone_tests/launcher_profile_schema.sh
launcher_profile_schema_ok
bash tests/standalone_tests/profile_validator.sh
profile_validator_ok
py_compile_pass files=17
```

`launcher.sh --print-config` was also run with the CMP50HX route and `fast` mode in a temporary, non-serving harness. It printed context 221184, max batched tokens 4096, max sequences 2, draft-pool concurrency 2, CUDA graph captures 8/16, and DFlash/7; no server was started. `git diff --check` passed.

### GP-21.3 production GPU acceptance carried into this release

Tested route: 2 × CMP50HX 20 GiB, SM75, PCIe Gen2 x16, TP2, Swift Qwen3.8-27B AWQ, DFlash2 K=7, target FP8 KV/draft FP16 KV, context 221184, `max_num_seqs=2`, Mamba align, CUDA Graphs, prefix caching, Custom AllReduce OFF.

| Prompt pair | TTFT | Output tokens | TG per request |
| --- | --- | --- | --- |
| 20,971 + 20,968 | 45.597 / 46.843 s | 1,353 / 1,494 | 58.34 / 59.47 tok/s |
| 32,012 + 32,011 | 70.37 / 71.22 s | 2 / 2 | NOT MEASURED |
| 64,004 + 64,003 | 147.44 / 149.35 s | 2 / 2 | NOT MEASURED |
| 95,996 + 95,995 | 245.92 / 246.69 s | 2 / 2 | NOT MEASURED |
| 199,990 + 16,000 | 346.38 / 344.96 s | 2 / 2 | NOT MEASURED |

Only the 21K pair performed substantial decode. The individual TG values sum to 117.81 tok/s, not a separately measured aggregate throughput. The earlier pair of ~21K prompts had TTFT 22.9 / 44.4 s and `running=1`, `waiting=1`; the candidate reached two active prefills, but its first-request TTFT was higher. No prefill speedup or comparable end-to-end throughput gain is established. The 32K–200K rows are admission/prefill checks with two output tokens each.

The reported cache vector changed from `[23,23,23,154,10,10,10,10,10]` to `[23,23,23,151,18,18,18,18,18]`. Draft pool capacity increased 80% per group; target attention capacity decreased about 1.95%; configured max context stayed 221184. The scalar reported KV capacity changed 252313→247398 tokens. A 7,980-token identical repeat reported 4,944 cached tokens.

### Production source comparison

Read-only SHA-256 comparison covered ten changed Python runtime files. Seven files matched byte-for-byte. `sched/output.py`, `metrics/loggers.py`, and `metrics/stats.py` differed only in comments/docstring formatting or the downstream modified-file notice; direct diffs confirmed executable code is the same. Exact hashes are in [`manifests/gp22-production-source-review.md`](manifests/gp22-production-source-review.md). No production configuration, service, or runtime was changed or restarted for GP-22.

### Security and package checks

- Gitleaks over all newly added commits from `origin/main` to the release candidate: **0 findings** (100% redacted output mode).
- Manual added-line scan for absolute user/host paths, private IPs/hostnames, credential markers, prompts, and internal mounts: **0 matches**.
- The published diff contains no production logs, model weights, virtual environments, CUDA binaries, caches, backups, or private launcher/service configuration.
- `README.md`, `README.ru.md`, `README.zh-CN.md`, `CHANGELOG.md`, `VERSION`, and the `pyproject.toml` fallback identify the downstream release consistently; inherited `PROJECT_RELEASE.env` runtime identity stays `0.2.2-post2`.

The clean build/install from the public GitHub checkout remains **NOT VERIFIED**. Full upstream tests, extended GPU endurance, full GDN lifecycle, GPU preemption/resume, CUDA Graph replay after slot reuse, HTTP client cancellation completion, and an apples-to-apples aggregate decode benchmark remain **NOT VERIFIED**. GPU data above comes from the earlier GP-21.3 production acceptance window; no benchmark or runtime restart was performed for this release task.

Historical runtime validation record date: 2026-11-15. This report separates runtime validation of the recovered Experimental B working tree from clean-build reproducibility of the published checkout and from GPU scenarios that remain open. Experimental B was exercised on the production two-CMP50HX host; it is incorrect to characterize the validation as CPU-only.

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

No GPU workload was started as part of GP-22 release verification. The recorded GP-12 and GP-21.3 runtime/GPU observations came from their earlier authorized validation windows; production was not restarted or modified for repository publication.

## Remaining regression coverage — not verified

- Clean native build/install directly from the published checkout on CMP50HX.
- GPU preemption/resume and cancellation-through-preemption interaction; cancellation itself was exercised, but latency/details were not captured in the published artifacts.
- Full GDN/Mamba slot lifecycle.
- CUDA Graph replay after scheduler-slot reuse. Graph capture/startup succeeded and graphs were active during runtime tests; replay-after-reuse remains distinct and unverified.
- Full long-duration acceptance beyond the recorded 30.5-minute agent run and 10-minute stream.
- Standardized PP/TG benchmark or performance comparison against Neith.
- Root cause of the 2026-11-10 host reboots and exact reboot-time source provenance. Reboot causation remains **UNKNOWN**; observed memory pressure is not proof.
