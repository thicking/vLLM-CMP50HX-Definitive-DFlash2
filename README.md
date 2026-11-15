<!-- Downstream changes for vLLM-CMP50HX-Definitive-DFlash2; see PATCHES.md. -->
# vLLM-CMP50HX-Definitive-DFlash2

Language: English | [Русский](README.ru.md)

An independently maintained source derivative of [vLLM](https://github.com/vllm-project/vllm) and [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive). This repository preserves the recovered Experimental B runtime snapshot and the associated GP-12 changes for SM75/CMP50HX serving. It is an experimental source release; it is not a new inference engine and does not claim upstream authorship.

## Release status

The project release identity is `v0.1.0-alpha.1`. It is independent from the Python package version (`0.2.3.dev0+g2d63e2e32.d20260928.cu130`) and upstream-fork version (`v0.2.2-post2`). No wheel or container image is published. A clean build and install from this repository have not been verified on CMP50HX.

The source snapshot is based on upstream-fork commit `2d63e2e322b7488776d1fb822f3556dbd7334116`. The production checkout was shallow and dirty. Its tracked diff was imported as one snapshot commit; the repository does not invent a patch-by-patch history. A SHA-256 manifest is in [`manifests/production-b-20261115.sha256`](manifests/production-b-20261115.sha256). Exact source state during the host reboots on 2026-11-10 is unavailable, so whether each recovered change existed then is **UNKNOWN**.

## Recovered runtime profile

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
| Logged KV capacity | 252313 tokens; about 1.14× one configured request context |

The shared KV allocator dynamically distributes blocks. Logged capacity is not a guarantee that two requests can each occupy their full 221184-token context at once. `MAX_NUM_SEQS=2` permits two scheduler slots; actual overlap also depends on request shape and available GDN/Mamba state. The portable route profile is [`profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env`](profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env). It contains route settings only, not private model paths, network configuration, or server launcher files.

## Recovered engine changes

The snapshot includes scheduler admission and capacity diagnostics; multi-sequence GDN/Mamba reservation accounting; DFlash2 restored-prefix accounting and one-time draft block-table shifting; allocator failure diagnostics; partial JSON streaming handling; Qwen3 structured required/named tool-choice support; and recovered SM75 DFlash2 model paths. File-level inventory and provenance are in [PATCHES.md](PATCHES.md). Existing CUDA Graph, DFlash2=7, quantization, TP=2, and parser configuration were retained. No speedup is attributed to individual changes.

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
