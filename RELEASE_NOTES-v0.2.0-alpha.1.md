# Workload-aware KV Cache & Parallel Inference

Experimental source prerelease: [`v0.2.0-alpha.1`](https://github.com/thicking/vLLM-CMP50HX-Definitive-DFlash2/releases/tag/v0.2.0-alpha.1).

## Highlights

- GP-20 adds live prefill telemetry: `vllm:prefill_computed_tokens_total`, `vllm:num_requests_prefilling`, and `vllm:prefill_tokens_remaining`. The computed-token counter reflects completed local prefill execution, not planned work; cache hits, external KV, decode, and speculative verification are excluded.
- GP-21.1 fixes a multi-waiter prefill-barrier livelock and bounds repeated scheduler diagnostics.
- GP-21.3 adds opt-in, workload-aware sizing of independent native DFlash2 draft KV pools.

## Scheduler correctness

An allocator-rejected FIFO head could be excluded from the alignment frontier while later, unreachable waiters still affected that frontier. The active prefill cohort then stopped progressing and the scheduler could emit an empty batch. The fix keeps later waiters out of that frontier and retries the blocked head after the active cohort drains. FIFO ordering, the prefill barrier, and `full_sequence_must_fit` remain enabled. This is separate from the upstream Mamba-alignment livelock fix.

## Workload-aware KV cache allocation

`--dflash-draft-pool-concurrency N` (profile key `DFLASH_DRAFT_POOL_CONCURRENCY=N`) requests a minimum per-draft-pool concurrency for native DFlash2 independent pools. `N` is an integer ≥1 and must be ≤ `max_num_seqs`; omitting it (`None`) preserves the previous proportional sizing. The allocator derives pool capacities from cache specs, physical page bytes, available memory, context and requested concurrency while preserving target Mamba reservations. An explicit user `num_gpu_blocks_override` is incompatible with this sizing mode. Temporary internal CUDA Graph profiling overrides remain supported.

Example addition to an already configured native DFlash2 serve command:

```bash
--dflash-draft-pool-concurrency 2
```

The reported scalar KV token capacity does not describe every independent hybrid cache-group constraint. The tested CMP50HX allocation is hardware/model-specific and should not be copied as a universal vector.

## Real-time prefill telemetry

The counter is incremented after a successful model-runner output and is available even when that step emits no output token. Recomputed prompt tokens after preemption count as actual execution. No additional CUDA synchronization is introduced, and existing cumulative metrics retain their semantics.

## Tested hardware and configuration

- 2 × NVIDIA CMP50HX 20 GiB, SM75, PCIe Gen2 x16; TP=2.
- Swift Qwen3.8-27B AWQ target; DFlash2 K=7 draft.
- Target KV FP8; draft KV FP16; context 221184; `max_num_seqs=2`.
- Mamba align, CUDA Graphs, prefix caching; Custom AllReduce OFF.

The allocation vector changed from `[23,23,23,154,10,10,10,10,10]` to `[23,23,23,151,18,18,18,18,18]`: each of five draft pools increased 80% (10→18 blocks), while target full-attention capacity decreased about 1.95% (154→151). Configured maximum context was unchanged. The reported KV capacity changed 252313→247398 tokens (−4915, ~1.95%). This scalar does not guarantee two fully occupied maximum-length sequences.

## GPU acceptance results

| Concurrent prompt tokens | TTFT per request | Output tokens | Notes |
| --- | --- | --- | --- |
| 20,971 + 20,968 | 45.597 / 46.843 s | 1,353 + 1,494 | TG 58.34 / 59.47 tok/s per request |
| 32,012 + 32,011 | 70.37 / 71.22 s | 2 + 2 | admission/prefill probe |
| 64,004 + 64,003 | 147.44 / 149.35 s | 2 + 2 | admission/prefill probe |
| 95,996 + 95,995 | 245.92 / 246.69 s | 2 + 2 | admission/prefill probe |
| 199,990 + 16,000 | 346.38 / 344.96 s | 2 + 2 | asymmetric admission/prefill probe |

The individual TG values for the 21K pair sum to 117.81 tok/s; that is not a separately measured aggregate throughput. The previous ~21K baseline had TTFT 22.9 / 44.4 s with `running=1`, `waiting=1`, and sequential prefill. The candidate confirms concurrent admission but has a higher first-request TTFT in that comparison. No prefill speedup or comparable total decode-throughput improvement is claimed. An identical 7,980-token prompt repeat reported 4,944 cached tokens.

## Configuration

Use the option only with an existing native DFlash2 route. The setting must not exceed scheduler `max_num_seqs`; it requires independently allocated DFlash draft groups. An explicit user `num_gpu_blocks_override` is rejected. The tested profile sets `MAX_NUM_SEQS=2` and `DFLASH_DRAFT_POOL_CONCURRENCY=2`.

## Validation

The focused CPU release subset passed: `9 passed, 305 deselected, 15 warnings in 25.44s`. The full upstream test suite and a clean native build from the public GitHub checkout were not run. Production source verification found seven byte-identical changed Python files and three files whose differences were limited to comments/formatting; see [TEST-REPORT.md](TEST-REPORT.md) and [the source SHA review](manifests/gp22-production-source-review.md).

## Known limitations

- Clean build/install directly from this published checkout: **NOT VERIFIED**.
- Long-duration GPU endurance and full GPU preemption/resume, GDN slot lifecycle, and CUDA Graph replay after slot reuse: **NOT VERIFIED**.
- Waiting-request cancellation was observed server-side, but completion of the HTTP client cancellation path: **NOT VERIFIED**.
- Long prompts beyond the ~21K pair generated two tokens per request; those tests are admission/prefill probes, not useful decode benchmarks.
- Comparable aggregate decode-throughput benchmark is unavailable; results are specific to the stated hardware and route.
- This is an alpha prerelease and does not claim universal production readiness.

## Upstream credits

This derivative preserves attribution and license terms for [vLLM](https://github.com/vllm-project/vllm), [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive), and third-party components. The minimal GP-21.1 patch and issue draft are included locally under [`upstream-proposals/`](upstream-proposals/) and have not been submitted upstream.

## Русская документация

Краткое описание и ограничения по-русски: [README.ru.md](README.ru.md).
