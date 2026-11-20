# Recovered source and downstream patch inventory

## Snapshot provenance

- Upstream-fork base commit: `2d63e2e322b7488776d1fb822f3556dbd7334116` (`release: publish v0.2.2 runtime identity`).
- Recovered Experimental B source: the production process working directory was checked through its systemd unit, active process argv, and `/proc/<pid>/cwd` on 2026-11-15. No runtime operation was performed for this import.
- The server checkout was shallow and dirty. Its tracked working-tree diff against the base was 32 files, 882 insertions and 100 deletions. Four new regression tests and a separate untracked `profiles/2xT10/` directory were also present.
- This repository imports those recovered changes as a single honest snapshot commit. Individual historical commits are not reconstructed. A complete source archive for the three 2026-11-10 reboot-time checkouts is unavailable; exact reboot-time contents and patch presence are **UNKNOWN**.
- A later repository commit adds Apache-2.0 modified-file notices to the 32 changed existing files and labels the four recovered regression tests. These are comments/documentation only and do not change runtime behavior. `VERSION` is a single-value input that rejects comments, so its notice is in `VERSION.notice` and referenced by `NOTICE`.
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

## GP-21.1 candidate changes (2026-11-20)

- `vllm/v1/core/sched/scheduler.py`: when an allocator-rejected FIFO waiter
  blocks admission, keep the active prefill cohort out of a frontier built
  from later waiters and stop retrying the known rejection in that scheduler
  step. FIFO admission and the existing prefill barrier remain enabled; the
  rejected head is retried when the active prefill cohort drains; cancellation
  removes its blocked-frontier state.
- `vllm/v1/core/gp12_diagnostics.py`, `scheduler.py`, and
  `kv_cache_manager.py`: bound GP-12 diagnostic state to at most 4096
  request/event keys per component, log the first rejection and state changes, summarize
  identical retries at exponentially increasing intervals, and clear state
  on admission, completion, cancellation, or KV release. This changes logging
  only, not allocator or scheduler decisions.
- `tests/v1/core/test_scheduler.py` and `test_prefix_caching.py` cover the
  no-progress reproduction, FIFO progress to completion, blocked-head
  cancellation, and independent per-group allocation vectors. The
  downstream-only `test_gp12_diagnostics.py` checks bounded state and log
  summaries. CPU checks do not establish GPU stability.

## GP-20 production telemetry compatibility (2026-11-20)

The production runtime already includes GP-20 live prefill telemetry. The
recovered B development base predates those production-only Python changes, so
this candidate carries the same telemetry contract forward to avoid replacing
it with an older scheduler during GP-21.1 deployment. The scheduler snapshots
locally computed prompt positions for the finalized batch and publishes the
counter only after a successful model-runner output; local prefix hits,
external KV, decode tokens, and speculative padding are excluded. The two
gauges report requests still pre-filling and their remaining prompt positions.
No scheduler admission or inference behavior is changed by this compatibility
portion. Its CPU regressions cover chunked prefill without output tokens,
mixed prefill/decode, prefix and external KV, preemption/resume, cancellation,
and Prometheus publication without iteration output.

## GP-21.3: workload-aware DFlash draft pools (2026-11-20)

- Adds opt-in `dflash_draft_pool_concurrency` /
  `--dflash-draft-pool-concurrency`; unset preserves proportional sizing.
  The active B profile opts in with `DFLASH_DRAFT_POOL_CONCURRENCY=2`.
- Marks native draft groups from their actual cache specs and sizes each
  independent DFlash pool from its `SlidingWindowSpec`, context, in-flight
  demand, and shared physical byte budget. Existing target Mamba reservations
  remain intact. Unsupported allocator overrides fail closed. Scheduler,
  admission, prefill barrier, DFlash algorithm, and cache grouping are unchanged.
- First production startup exposed a guard-placement bug: CUDA Graph profiling
  temporarily sets `num_gpu_blocks_override` for a minimum cache. The generic
  low-level builder mistook this internal profiling override for a user
  override. The guard was moved to the real `get_kv_cache_configs()` sizing
  path, preserving rejection of an explicit user override. Targeted tests
  passed, and the corrected candidate then started successfully on production.
- Production allocator log (per TP rank): cache capacity `247,398` tokens;
  concurrency `1.12x` for `max_model_len=221184`; total backing `5.82 GiB/rank`.
  Actual groups were:

  | Groups | Role/spec | Capacity / usable / minimum blocks | Page bytes | Backing bytes per group |
  |---|---|---:|---:|---:|
  | 0–2 | target `MambaSpec` | 23 / 22 / 20 | 27,000,832 | 621,019,136 |
  | 3 | target `FullAttentionSpec` | 151 / 150 / 135 | 27,000,832 | 4,077,125,632 |
  | 4–8 | draft `SlidingWindowSpec` | 18 / 17 / 16 | 3,375,104 | 60,751,872 |

  `usable_blocks` excludes the null page. Target attention lost three blocks
  versus the original vector `[23,23,23,154,10,10,10,10,10]`; each of five
  draft pools gained eight. The maximum context remains configured at 221184.
  This is not a guarantee that two full-context sequences fit simultaneously.
- GPU acceptance on two CMP50HX GPUs: simultaneous pairs completed at ~21K,
  32K, 64K, and 96K prompt sizes; an asymmetric `199990 + 16000` pair also
  reached `running=2` and `prefilling=2` without allocator rejection. The
  asymmetric pair retained context 221184 and completed both short outputs.
  The ~21K pair produced 1,353 / 1,494 output tokens with TTFT 45.597 / 46.843
  s and TG 58.34 / 59.47 tok/s. Longer-prompt admission probes used two output
  tokens each; they validate admission/prefill, not useful decode throughput.
- Prefix reuse was confirmed with an 8K identical repeat: `cached_tokens=4944`
  (three 1648-token cache blocks); computed-prefill delta matched the uncached
  prompt remainder. A waiting cancellation probe cleared the waiting slot
  within about 1.54 s while both active requests continued; the client harness
  reported a socket-close `AttributeError`, so client-side cancellation
  completion is not claimed from that probe. Earlier CPU lifecycle tests cover
  allocator release, cancellation and prefix-cache preemption.
- Post-restart smoke: `/health=200`, model `swift15-qwen38-awq`, context
  `221184`, service active, `NRestarts=0`, and queue gauges returned to zero.
  A post-restart smoke request returned HTTP 200 with a short prompt and
  response; DFlash counters advanced. Startup arguments confirmed TP2,
  DFlash K=7, draft FP16 KV, target FP8 KV, prefix caching, CUDA Graphs,
  `qwen3_xml`, and Custom AllReduce OFF. No matching CUDA/NCCL/Xid/AER/OOM
  errors appeared in the inspected runtime and kernel logs after restart.
- Targeted CPU test for the per-group diagnostic log:
  `1 passed, 119 deselected, 15 warnings in 8.81s`. Earlier candidate CPU
  lifecycle/sizing subset: `5 passed, 14 warnings in 13.78s`. Profile and
  launcher checks were green before deployment. No full upstream suite or
  CUDA/native extension rebuild was run.
- Rollback artifacts and numeric-only acceptance telemetry remain on the
  private test host and are excluded from this source release. No prompts or
  generated content are part of the published tree.

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
