<!-- Downstream changes for vLLM-CMP50HX-Definitive-DFlash2; see PATCHES.md. -->
# Changelog

This changelog tracks releases of vLLM 2080 Ti Definitive Edition separately from upstream vLLM releases.

## v0.2.0-alpha.2 - 2026-11-22 (documentation-only downstream release)

- Publishes GP-24/GP-25 operational findings on Prefix Cache, hybrid KV residency, tool-schema stability, matched A/B measurements, and resource/toolchain lessons.
- Adds no engine, profile, launcher, dependency, model, or production configuration changes and includes no new GPU test results.
- The detailed runtime evidence and limitations remain in `TEST-REPORT.md`; clean build/install from the public checkout remains NOT VERIFIED.

## v0.2.0-alpha.1 - 2026-11-20 (independent downstream project)

**Workload-aware KV Cache & Parallel Inference** — experimental source
prerelease for the validated dual-CMP50HX route.

- Adds GP-20 live prefill-compute telemetry, published only after successful
  execution steps and excluding cached/external tokens, decode, and speculative
  verification.
- Fixes the GP-21.1 multi-waiter prefill-barrier livelock while preserving FIFO
  admission, the barrier, and full-sequence allocator checks; bounds scheduler
  diagnostic state and repeated logs.
- Adds opt-in GP-21.3 workload-aware DFlash2 draft-pool sizing based on cache
  specifications, page sizes, available memory, and requested concurrency.
- Includes target CPU regression tests, the two-CMP50HX acceptance record, and
  a separately prepared, unpublished upstream proposal.
- Remains a prerelease: clean build from the public checkout, complete GPU
  lifecycle/endurance, HTTP cancellation completion, and comparable decode
  throughput are not verified. No wheel or container image is published.
- The repository tag is independently versioned; the inherited runtime
  manager identity remains `v0.2.2-post2`.

## v0.1.0-alpha.1 - 2026-11-15 (independent downstream project)

- Publishes the recovered Experimental B source snapshot for the CMP50HX route
  as an independently versioned downstream project.
- Adds the 2xCMP50HX Qwen3.8-27B AWQ DFlash2=7 route profile and records the
  recovered runtime settings and evidence limits.
- Preserves the upstream-fork commit history and adds the recovered dirty-tree
  changes as a single snapshot commit; see `PATCHES.md` for provenance.
- This project release tag is separate from the upstream-fork and Python
  package versions. It does not claim a clean build or complete GPU acceptance.

## v0.2.2-post2 - 2026-09-26

Patch release following `v0.2.2-post1`. This release further strengthens
DFlash2 prefix-cache replay and target-state handling.

### DFlash2 prefix cache and target state

- Retains DFlash2 target replay state across prefix-cache reuse and alternating
  prefill/decode requests, so reusable target state is not lost between turns.
- Restores the configured target Mamba speculative state pages for DFlash2 and
  DSpark while keeping draft KV pages in their independent pools.
- Accounts for the resident replay and hashed-boundary pages together with the
  target speculative pages in Mamba align mode, keeping target block tables and
  KV capacity consistent during speculative replay.
- Adds regression coverage for prefix-cache replay and DFlash target state
  sizing and accounting.

## v0.2.2-post1 - 2026-09-26

Patch release following `v0.2.2`, based on the latest `main` runtime.

### DFlash and hybrid Mamba correctness

- Keeps requests that already emitted decode tokens active while a later request
  continues chunked prefill, preventing unnecessary decode pauses.
- Stops reserving speculative Mamba state pages for DFlash2 and DSpark target
  KV, while retaining the resident replay state required by align mode.
- Preserves hashed Mamba align boundary state across decode and releases only
  obsolete unhashed blocks, keeping DFlash target prefix-cache entries reusable.
- Preserves restored-token accounting through the following DFlash decode step.

### Profiles and validation

- Adds four validated 2xT10 Qwen3.8-27B INT4 W4A16 MTP3 routes covering FP8 KV,
  TQ4NC, TQK8V4, text-only, and text-image serving.
- Adds focused scheduler, KV sizing, and prefix-cache regression coverage for
  the repaired paths.

## v0.2.2 - 2026-09-25

Incremental release from the validated `v0.2.1` SM75 serving snapshot
(`b296055efd`) through the current `main` branch. This release focuses on
correctness and reproducibility for Qwen3.8 hybrid Mamba serving on Turing
GPUs, and introduces the published runtime identity `vllm-def-cu130`.

### Prefix caching and speculative decoding

- Reuses DFlash2 target KV across prefix-cache hits and retires replaced
  hashed Mamba states at aligned prefill boundaries.
- Fixes FP8 KV hybrid prefix hits when the Mamba block size equals the hash
  block size, including the 1648-token SM75 route.
- Preserves request-row boundaries and hardens FULL-graph padding during
  speculative replay.
- Keeps SM75 DFlash2 tensor-parallel reductions in one scale domain.

### MTP, GDN, and TurboQuant correctness

- Prevents late prefills from pausing active MTP decode and preserves exact
  verifier KV when a late prefill joins an MTP batch.
- Fixes the hybrid-Mamba prefill barrier livelock and adds explicit alignment
  limits and tail-cohort coverage.
- Stabilizes TurboQuant verifier logits across late prefills and fails closed
  when adaptive query boundaries cannot be honored.
- Partitions stateful Qwen GDN forward execution and fixes SM75 GDN/DFlash
  speculative quality regressions.

### SM75 runtime and launcher

- Prevents unsupported CUTLASS FP8 dispatch on SM75 by selecting the supported
  fallback path.
- Adds safer FULL-graph padding and trace controls for the SM75 runtime.
- Makes the launcher load the model's generation configuration by default.
- Synchronizes renamed validated profile references and refreshes the shipped
- Adds the validated 4xT10 W8A16 FP8KV DFlash2 text-only route at 2x220K.
- Keeps FlashInfer MTP draft metadata correct across metadata rebuilds and
  preserves eager MTP execution for the SM75 compatibility path.

## v0.2.1 - 2026-09-19

Stable release for the CUDA 13 / PyTorch 2.13 SM75 line.

- Promotes the validated SM75 serving routes and profile set from the RC.

## v0.2.1-RC1 - 2026-09-18

Release candidate for the CUDA 13 / PyTorch 2.13 SM75 line.

- Adds validated DFlash2 routes, fused Triton grouped convolution, stable SM75
  FlashInfer FA2 graph buffers, and live metadata refresh for FULL graph replay.
- Fixes concurrent speculative graph sizing, automatic prefill alignment,
  independent target/draft KV pools, and concurrent MTP FA2 execution.
- Generalizes qualified PCIe custom all-reduce to 2–16 GPUs and improves the
  CUDA 13 Torch/Triton mirror build path.
- Converges profiles on validated routes, requires explicit KV precision, and
  removes unsupported concurrency presets.
- Moves profiles to a flat hardware/model/weight route layout and makes `fast`
  the launcher default startup mode.

## v0.2.1-pre4 - 2026-09-14

Nightly-alignment candidate based on upstream `b23433088b`
(`v0.29.1rc0-33`).

- Adopts upstream replacements for previously local runtime fixes and retains
  only the SM75-specific FlashQLA, TurboQuant, parser, and rendezvous paths.
- Keeps `humming-kernels[cu13]==0.1.13` for INT6/AutoRound loading.

## v0.2.1-pre3 - 2026-08-27

Validated true-concurrency candidate for dual RTX 2080 Ti.

- Adds packed-varlen FlashQLA, prefill batch alignment, bounded TurboQuant
  wrapper caching, and long-context workspace handling.
- Improves reasoning/tool parsing, Mamba/offload boundaries, GPU selection, and
  single-node process rendezvous.
- Validates the documented NVFP4 concurrency route and the associated parser,
  serving, TurboQuant, offload, and FlashQLA paths.

## v0.2.1-pre2 - 2026-08-21

First validated preview based on upstream vLLM `v0.27.1`.

- Adds Qwen3.8 FP8/NVFP4 profiles, TP/PP launcher selection, stricter preflight
  checks, and the required SM75 FlashInfer/FlashQLA compatibility paths.
- Ports the validated reasoning, tool-call, CUDA Graph, and TurboQuant fixes to
  the newer runtime architecture.
- Validates the documented dual-2080-Ti TP=2 routes; broader PP and additional
  long-context routes remained experimental.

## v0.2.1-pre

Initial CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 migration snapshot based
on upstream vLLM `v0.27.1`.

Thanks to upstream vLLM and its contributors. Fork release and validation work
is maintained by [@weicj](https://github.com/weicj), with contributions from
the project community.
