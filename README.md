<!-- Downstream changes for vLLM-CMP50HX-Definitive-DFlash2; see PATCHES.md. -->
# vLLM-CMP50HX-Definitive-DFlash2

Language: English | [Русский](README.ru.md)

An independently maintained source derivative of [vLLM](https://github.com/vllm-project/vllm) and [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive). This repository preserves the recovered Experimental B runtime snapshot and downstream GP-12, GP-20, GP-21.1, and GP-21.3 changes for SM75/CMP50HX serving. It is an experimental source release; it is not a new inference engine and does not claim upstream authorship.

## Release status

The latest downstream project release is [`v0.2.0-alpha.2`](https://github.com/thicking/vLLM-CMP50HX-Definitive-DFlash2/releases/tag/v0.2.0-alpha.2), a documentation-only supplement. The runtime source prerelease remains [`v0.2.0-alpha.1`](https://github.com/thicking/vLLM-CMP50HX-Definitive-DFlash2/releases/tag/v0.2.0-alpha.1), titled **Workload-aware KV Cache & Parallel Inference**; the inherited upstream-fork source identity remains `v0.2.2-post2`. No engine source, profile, launcher, model or production setting changes in alpha.2. No wheel or container image is published. A clean build and install from this repository have not been verified on CMP50HX.

Operational findings: [Prefix Cache guide](docs/prefix-cache.md), [benchmarks](docs/benchmarks.md), and [Russian lessons learned](docs/lessons-learned.ru.md).

The source snapshot is based on upstream-fork commit `2d63e2e322b7488776d1fb822f3556dbd7334116`. The production checkout was shallow and dirty. Its tracked diff was imported as one snapshot commit; the repository does not invent a patch-by-patch history. A SHA-256 manifest for that recovered snapshot is in [`manifests/production-b-20261115.sha256`](manifests/production-b-20261115.sha256). Exact source state during the host reboots on 2026-11-10 is unavailable, so whether each recovered change existed then is **UNKNOWN**.

## Validated workload-aware runtime profile (GP-21.3)

| Setting | Observed configuration |
| --- | --- |
| Hardware | 2 × NVIDIA CMP 50HX 20 GB, Turing TU102 / SM75 |
| Target | Swift Qwen3.8-27B AWQ, `compressed-tensors` |
| Tensor parallelism | TP=2 on GPU0/1 |
| Draft | DFlash2 GPTQ INT8, TP=2 |
| Target / draft KV | FP8 / FP16 |
| `MAX_MODEL_LEN` | 221184 tokens per request |
| `MAX_NUM_SEQS` | 2 scheduler sequences |
| DFlash tokens | 7 |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes 8 and 16; DFlash full graphs enabled |
| Prefix caching | Enabled; Mamba cache mode `align` |
| Custom AllReduce | Disabled |
| GPU utilization / max batched tokens | 0.986 / 4096 |
| Logged KV capacity after GP-21.3 | 247398 tokens; about 1.12× one configured request context |

The shared KV allocator dynamically distributes blocks. Logged capacity is not a guarantee that two requests can each occupy their full 221184-token context at once. `MAX_NUM_SEQS=2` permits two scheduler slots; actual overlap also depends on request shape and available GDN/Mamba state. The portable route profile is [`profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env`](profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env). It contains route settings only, not private model paths, network configuration, or server launcher files.

## Recovered engine changes

The snapshot includes scheduler admission and capacity diagnostics; multi-sequence GDN/Mamba reservation accounting; DFlash2 restored-prefix accounting and one-time draft block-table shifting; allocator failure diagnostics; partial JSON streaming handling; Qwen3 structured required/named tool-choice support; and recovered SM75 DFlash2 model paths. File-level inventory and provenance are in [PATCHES.md](PATCHES.md). Existing CUDA Graph, DFlash2=7, quantization, TP=2, and parser configuration were retained. No speedup is attributed to individual changes.

## v0.2.0-alpha.1 runtime changes

- **Live prefill telemetry (GP-20):** adds a cumulative counter exposed as `vllm:prefill_computed_tokens_total`, plus `vllm:num_requests_prefilling` and `vllm:prefill_tokens_remaining`. The counter advances only for successfully completed local prefill work. Prefix-cache and external-KV tokens, decode, and speculative verification are excluded; completed recomputation after preemption is counted as actual work. Existing prompt-token counters retain their completion-oriented semantics.
- **Prefill barrier correctness (GP-21.1):** prevents a later FIFO waiter, which cannot yet be admitted, from extending the alignment frontier after an earlier waiter is rejected by KV allocation. The active prefill cohort can continue; the rejected head is retried after that cohort drains. The fix keeps the prefill barrier and `full_sequence_must_fit` behavior.
- **Bounded scheduler diagnostics (GP-21.1):** bounds per-request diagnostic state, suppresses unchanged repeated events with backoff, and emits state changes/recovery. It does not change scheduling decisions.
- **Workload-aware DFlash2 pools (GP-21.3):** adds opt-in `dflash_draft_pool_concurrency`. When set, the allocator uses actual cache specs, physical page sizes, available memory, and requested draft concurrency to size independent draft pools while preserving target Mamba reservations. The default `None` keeps the prior proportional sizing.

### DFlash draft-pool option

The exact vLLM CLI option is `--dflash-draft-pool-concurrency N`; the profile/launcher environment key is `DFLASH_DRAFT_POOL_CONCURRENCY=N`. `N` must be an integer ≥1 and no greater than scheduler `max_num_seqs`. It applies only when native DFlash2 independent KV pools and identifiable DFlash `SlidingWindowSpec` groups are present. Omit it to retain the previous sizing. An explicit user `num_gpu_blocks_override` cannot be combined with this workload-aware plan. The allocator validates that combination at real startup sizing; CUDA Graph profiling's temporary internal override remains supported.

For an existing native DFlash2 `vllm serve` command, add:

```bash
--dflash-draft-pool-concurrency 2
```

The matching profile form is:

```dotenv
MAX_NUM_SEQS=2
DFLASH_DRAFT_POOL_CONCURRENCY=2
```

The number sizes the draft pools; it does not raise the scheduler's sequence limit. Allocation is model/cache-layout and hardware dependent. A scalar KV-token headline cannot express the independent capacities, page sizes, alignment requirements, and reservations of hybrid Mamba/attention/draft groups. The measured CMP50HX vector below is specific to the tested route and is not a universal preset.

### GP-21.3 GPU acceptance

On the tested two-CMP50HX route, the reported group capacities changed from `[23,23,23,154,10,10,10,10,10]` to `[23,23,23,151,18,18,18,18,18]`. Each of five draft pools gained 80% capacity (10→18 blocks); target full-attention capacity fell about 1.95% (154→151). The configured per-request context stayed 221184. The scalar reported KV capacity changed from 252313 to 247398 tokens (−4915, about −1.95%); this scalar is not a guarantee that two full-context requests fit simultaneously.

| Concurrent prompt tokens | TTFT per request |
| --- | --- |
| 20,971 + 20,968 | 45.597 / 46.843 s |
| 32,012 + 32,011 | 70.37 / 71.22 s |
| 64,004 + 64,003 | 147.44 / 149.35 s |
| 95,996 + 95,995 | 245.92 / 246.69 s |
| 199,990 + 16,000 | 346.38 / 344.96 s |

The approximately 21K pair generated 1,353 / 1,494 tokens at 58.34 / 59.47 tok/s per request. Their individual rates sum to 117.81 tok/s, which is **not** a separately measured aggregate throughput. Other long-prompt pairs generated two tokens per request and validate concurrent admission/prefill only. The earlier two-request baseline at about 21K reported TTFT 22.9 / 44.4 s with `running=1`, `waiting=1` and sequential prefill. The candidate confirms two-query overlap but has a higher first-request TTFT in that comparison; these runs do not establish a prefill speedup or a comparable total decode-throughput gain. An identical 7,980-token repeat reported 4,944 cached tokens. Full conditions and limitations are in [TEST-REPORT.md](TEST-REPORT.md).

### Scheduler livelock diagnosis

The reduced CPU reproducer exposed a FIFO/frontier livelock: the allocator rejects the queue head, a blocked marker removes it from the alignment frontier, a later unreachable waiter still affects the frontier, the active prefill cannot advance, and the scheduler emits an empty batch. The regression was reproduced on Definitive v0.2.2, post2, and post3 source lines. The patch prevents frontier promotion past the rejected head while the active cohort is running; it does not disable the barrier or `full_sequence_must_fit`. This is separate from the earlier Mamba-alignment livelock fix inherited from the upstream fork. A proposed upstream patch and issue draft are provided under [`upstream-proposals/`](upstream-proposals/); neither has been submitted.

## Evidence and limitations

### Runtime validation on CMP50HX

The recovered Experimental B was exercised on the actual two-GPU CMP50HX host; it is not an engine that has only received CPU or source checks. GP-12 included production runtime use and bounded GPU acceptance under the recorded profiles below. Results establish the tested paths and conditions, not every workload or a general stability guarantee.

- **10-minute stream and concurrent request (2026-11-14):** at a 147456-token per-request context, a streaming request returned HTTP 200 and ran for the 600-second client limit after first token (TTFT 1.31 s; 12,513 chunks, 240,939 characters). During the stream, an uneven 2,364-token prompt / 17-token output request returned HTTP 200 in 3.46 s. The scheduler log places both IDs in one batch with `running=2`; this confirms actual overlap for this pair. DFlash counters reached 13,025 drafts / 33,161 accepted. The logged KV capacity was 229,376 tokens. The tested pair fit dynamically; two requests each filling the full context were not tested and would exceed that logged capacity. 147456 is the highest sampled passing context for this pair; the sampled 160K pair did not fit, so this is not a universal maximum.
- **Post-reboot current-profile acceptance (2026-11-15):** B restarted on the two GPUs with context 221184, `MAX_NUM_SEQS=2`, DFlash2=7, prefix caching, CUDA Graphs, and `qwen3_xml`. A simultaneous text stream and `lookup_city` tool request both returned HTTP 200; the scheduler reported `running=2`, the tool JSON was valid, and DFlash counters increased. A repeated identical 20,925-token prompt with a named tool completed three times (25.793 / 4.541 / 3.343 s); prefix hits advanced 0 → 18,128 → 36,256. This confirms the exercised GPU prefix-restore/repeated-admission path, not every partial-block or preemption mapping. An uneven approximately 190,126-content-token prompt plus a short tool request also passed (KV peak 77.27%; exact full chat-template token count was not captured).
- **Agent endurance (2026-11-15):** an isolated coding-agent task completed 28 cycles, 84 tool calls, 28 edits and 56 reads over 1,828.9 seconds (~30.5 minutes). Health remained true; service PID stayed stable; no guard or runtime error was recorded. Across 177 samples, GPU0/1 reached 100% utilization and maximum temperatures were 60/69 °C; GPU2 remained idle. Minimum `MemAvailable` was 5.601 GiB and `SwapFree` 7.523 GiB; memory PSI `avg10` peaked at 0. This is a bounded run, not multi-hour or multi-day endurance.
- **Cancellation and client use:** cancellation was exercised during GP-12 without a reported stuck runtime or service failure. OpenCode and Open WebUI client workflows were also exercised against B. These are owner-confirmed acceptance observations, not a version-complete client compatibility matrix; cancellation timing and UI-version details were not captured in the published test artifacts.
- **Current operation:** as of 2026-11-15, production `vllm-b.service` was active on GPU0/1, `/health` returned 200, and `/v1/models` reported `swift15-qwen38-awq` with context 221184. The owner reports seeing around **90 tok/s on average** during real tasks and stable operation. Treat this as a preliminary live-use observation: it is not a controlled benchmark and has no fixed prompt, context, concurrency, or measurement window.

One B-only throughput snapshot measured about 41.76 tok/s for a single 256-token cold-prefix output and estimated 52.85 / 49.55 tok/s for two simultaneous 128-token outputs. These short per-request observations are configuration-specific and are not directly comparable with the owner's ~90 tok/s live-use observation. No Neith comparison or general speedup is claimed.

### Clean-build reproducibility and remaining GPU coverage

The production runtime evidence above came from the recovered Experimental B working tree. A **clean native build and install from the published GitHub checkout have not been reproduced**. This is a separate packaging/build question and does not invalidate the runtime tests of the recovered build. No wheel or container image is published.

Startup graph capture was observed, and CUDA Graphs were active during runtime acceptance. **CUDA Graph replay after slot reuse** is still not independently covered. Full GDN/Mamba slot lifecycle and GPU preemption/resume remain incomplete; cancellation acceptance does not imply preemption/resume coverage. The CPU cache-shift regression reproduced baseline FAIL and patched PASS, and GPU prefix restore was exercised in the repeated-prompt case, but this does not cover every GPU cache-shift lifecycle branch. Required tool calling with default reasoning also remains a known parser limitation.

Known open items:

- The cause of three reported host reboots on 2026-11-10 remains **UNKNOWN**. RAM/swap pressure was observed, but causation is unproven.
- The DFlash2 cache-shift defect is not established as a cause of those reboots; its presence in the reboot-time source is also **UNKNOWN**.
- GPU-level preemption/resume, complete GDN slot lifecycle, and CUDA Graph replay after slot reuse still need direct acceptance.
- `tool_choice=required` combined with reasoning remains unstable.
- CPU regression tests do not replace the remaining GPU lifecycle tests; a clean build from this GitHub checkout has not been verified.
- The ~90 tok/s figure is owner-reported operational observation, not a formal benchmark; no performance comparison against Neith or another runtime is claimed.

## Build and checks

The observed runtime environment included Python 3.12.14, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1, `compressed-tensors` 0.17.0, Transformers 5.17.0, and NVIDIA driver 610.43.03. These are environment observations, not a complete lockfile. Runtime NCCL library version and `nvcc` availability were not verified. The production virtual environment did not contain `pip`.

The repository retains `build.sh`, `PROJECT_RELEASE.env`, and `requirements/build/`, but a clean native build is **NOT VERIFIED**. Use an isolated build host and retain the complete build log. Do not copy production virtual environments, weights, runtime caches, or private paths into this repository.

Focused CPU regression checks and static/profile validation are recorded in [TEST-REPORT.md](TEST-REPORT.md). The profile validator checks profile structure, not GPU capacity or runtime startup.

## Attribution and license

The project-level license is Apache-2.0. Original vLLM, weicj, and third-party authors retain their copyright and authorship. Upstream license, attribution, SPDX, and acknowledgement material is preserved. The original credits section is retained verbatim in [UPSTREAM-CREDITS.md](UPSTREAM-CREDITS.md). See [LICENSE](LICENSE), [NOTICE](NOTICE), [VERSION.notice](VERSION.notice), and [PATCHES.md](PATCHES.md). Third-party components remain subject to their own terms. The CMP50HX unlock is maintained separately by [xrip/cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock); GPU unlock and driver modifications are outside this project.

See [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), and [CHANGELOG.md](CHANGELOG.md). Upstream changes are reviewed manually; there is no automatic merge into this project's `main`.
