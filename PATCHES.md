# Recovered source and downstream patch inventory

## Snapshot provenance

- Upstream-fork base commit: `2d63e2e322b7488776d1fb822f3556dbd7334116` (`release: publish v0.2.2 runtime identity`).
- Recovered Experimental B source: the production process working directory was checked through its systemd unit, active process argv, and `/proc/<pid>/cwd` on 2026-11-15. No runtime operation was performed for this import.
- The server checkout was shallow and dirty. Its tracked working-tree diff against the base was 32 files, 882 insertions and 100 deletions. Four new regression tests and a separate untracked `profiles/2xT10/` directory were also present.
- This repository imports those recovered changes as a single honest snapshot commit. Individual historical commits are not reconstructed. A complete source archive for the three 2026-11-10 reboot-time checkouts is unavailable; exact reboot-time contents and patch presence are **UNKNOWN**.
- The initial recovered `profiles/2xT10/` files were untracked in the production checkout. They are retained here, not attributed to this downstream project.

## Runtime and engine changes

| File(s) | Recovered behavior | Provenance status |
| --- | --- | --- |
| `vllm/v1/core/sched/scheduler.py` | Permit an already-decoding request to continue while a later request is in prefill; add scheduler admission/running/waiting/capacity diagnostics. | Concurrency and diagnostics work is GP-12. Presence during Nov 10 reboots is UNKNOWN. |
| `vllm/v1/kv_cache_interface.py` | Compute Mamba/DFlash/DSpark state reservation for multiple sequences, including resident state, verifier state, prefill checkpoints and prefix-hit copy-on-write. | GP-12. The reservation changes capacity accounting; they do not claim allocator speedup. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | Retain hashed align-mode Mamba blocks until safe release. | Present in the recovered dirty tree; whether it predates GP-12 or the original snapshot is UNKNOWN. |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | Preserve target-only restored-prefix token count and shift newly gathered draft block tables once per scheduler step; reset state on new admission. | GP-12 cache-shift correction. Its relation to the Nov 10 reboot cause is unproven. |
| `vllm/v1/core/kv_cache_manager.py` | Log required/free/capacity/watermark/reserved block details for allocator failures. | GP-12 diagnostics; no allocation-policy or speed claim. |
| `vllm/tool_parsers/streaming.py` | Treat `IndexError` while parsing incomplete partial JSON as an incomplete fragment. | GP-12. Malformed arguments are not promoted to executable calls by this change. |
| `vllm/tool_parsers/qwen3_engine_tool_parser.py` | Enable structured required/named tool-choice path for Qwen3. | GP-12 change after the earlier stage-one snapshot. `tool_choice=required` with reasoning remains unstable. |
| `vllm/entrypoints/generate/base/protocol.py`, `vllm/entrypoints/generate/base/serving.py`, `vllm/v1/metrics/stats.py` | Add serving protocol and per-request speculative metric support used by the recovered runtime. | Present in the recovered dirty tree; GP-12 attribution is mixed/uncertain. |
| `vllm/model_executor/models/qwen3_dflash.py`, `vllm/model_executor/models/qwen3_dflash2.py`, `vllm/model_executor/models/dflash2_sm75.py` | Preserve GPTQ/Marlin projection compatibility, SM75 DFlash2 kernels and the recovered Turing-specific MLP path. | At least part of this work is present in Nov 10 backups and predates GP-12. No performance attribution is made here. |
| `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_5_mtp.py` | Preserve recovered Qwen 3.5/MTP source changes. | Present before this import; exact author, date and relationship to the reboot-time trees are UNKNOWN. |

## Regression tests

New tests imported from the recovered working tree:

- `tests/parser/engine/test_qwen3_required_tool_choice.py`
- `tests/parser/engine/test_required_tool_streaming.py`
- `tests/v1/core/test_dflash_mamba_concurrency_capacity.py`
- `tests/v1/spec_decode/test_dflash_cache_shift_lifecycle_cpu.py`

The cache-shift CPU test previously reproduced a failure on the unpatched source and passed after the patch; it covers restored prefix, the next scheduler step, block mapping, preemption/resume, slot reuse, and ordinary decode. This is CPU-level evidence, not GPU lifecycle acceptance.

Existing test files changed in the recovered tree: `tests/entrypoints/openai/chat_completion/test_serving_chat.py`, `tests/v1/core/test_prefix_caching.py`, `tests/v1/core/test_scheduler.py`, `tests/v1/core/test_single_type_kv_cache_manager.py`, `tests/v1/metrics/test_stats.py`, and `tests/v1/spec_decode/test_dflash2.py`.

## Launcher, build, profiles, and release metadata

Recovered dirty-tree changes include `launcher.sh`, `build.sh`, `PROJECT_RELEASE.env`, `VERSION`, `pyproject.toml`, and `tools/validate_profiles.sh`. The raw `SPECULATIVE_CONFIG` / `MTP_K=0` branch is visible in a preserved Nov 10 launcher copy; other launcher/build changes have mixed or unknown provenance. They are imported without claiming that every line was added during GP-12.

The recovered B route is documented in `profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env`: context 221184, sequence limit 2, FP8 target KV, DFlash/7, GPU util 0.986 and max batched tokens 4096. This is a configuration export, not a promise of two full contexts. Four untracked 2xT10 profiles and their hardware notes were also preserved from the recovered source tree.

The existing upstream changelog/history is retained. The independent Git release tag is `v0.1.0-alpha.1`; the Python package version and the upstream-fork version are unchanged by that tag.

## Checks and unresolved items

- Critical production-source SHA-256 hashes for scheduler, KV interface/manager, DFlash speculator, parser, model implementations, and all four new tests matched the read-only source copy on 2026-11-15.
- CPU regression tests and static checks are reported in the release record; GPU tests are not run as part of repository creation.
- Historical reboot root cause and exact reboot-time patch provenance remain UNKNOWN.
- Direct preemption/resume, full GDN slot lifecycle, and CUDA Graph replay after slot reuse remain incomplete.
- No speedup over Neith or a benchmark guarantee is claimed.
