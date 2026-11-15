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

Recorded runtime observations include two requests visible simultaneously as `running=2`, short overlapping streaming and tool-call requests, prefix reuse, and one isolated 30-minute coding-agent run (28 completed cycles and 84 tool calls) without a recorded runtime error or service restart. These are bounded acceptance observations, not an endurance guarantee. No PP/TG figures are published because metric provenance and semantics have not been independently validated for a benchmark. The full GPU lifecycle suite is incomplete.

Known open items:

- The cause of three reported host reboots on 2026-11-10 remains **UNKNOWN**. RAM/swap pressure was observed, but causation is unproven.
- The DFlash2 cache-shift defect is not established as a cause of those reboots; its presence in the reboot-time source is also **UNKNOWN**.
- GPU-level preemption/resume, complete GDN slot lifecycle, and CUDA Graph replay after slot reuse still need direct acceptance.
- `tool_choice=required` combined with reasoning remains unstable.
- CPU regression tests do not replace GPU lifecycle tests; a clean build has not been verified.
- No performance comparison against Neith or other runtime is claimed.

## Build and checks

The observed runtime environment included Python 3.12.14, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1, `compressed-tensors` 0.17.0, Transformers 5.17.0, and NVIDIA driver 610.43.03. These are environment observations, not a complete lockfile. Runtime NCCL library version and `nvcc` availability were not verified. The production virtual environment did not contain `pip`.

The repository retains `build.sh`, `PROJECT_RELEASE.env`, and `requirements/build/`, but a clean native build is **NOT VERIFIED**. Use an isolated build host and retain the complete build log. Do not copy production virtual environments, weights, runtime caches, or private paths into this repository.

Focused CPU regression checks and static/profile validation are recorded in [TEST-REPORT.md](TEST-REPORT.md). The profile validator checks profile structure, not GPU capacity or runtime startup.

## Attribution and license

The project-level license is Apache-2.0. Original vLLM, weicj, and third-party authors retain their copyright and authorship. Upstream license, attribution, SPDX, and acknowledgement material is preserved. The original credits section is retained verbatim in [UPSTREAM-CREDITS.md](UPSTREAM-CREDITS.md). See [LICENSE](LICENSE), [NOTICE](NOTICE), [VERSION.notice](VERSION.notice), and [PATCHES.md](PATCHES.md). Third-party components remain subject to their own terms. The CMP50HX unlock is maintained separately by [xrip/cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock); GPU unlock and driver modifications are outside this project.

See [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), and [CHANGELOG.md](CHANGELOG.md). Upstream changes are reviewed manually; there is no automatic merge into this project's `main`.
