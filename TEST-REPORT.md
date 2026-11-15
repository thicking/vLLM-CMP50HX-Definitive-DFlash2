# Recovered Experimental B validation record

Record date: 2026-11-15. This report describes the imported working-tree snapshot and the validation available for the independent source publication. It is not a claim of full GPU acceptance.

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

The exact cache-shift lifecycle test had previously failed on the unpatched source and passed after the narrow fix. It is CPU evidence; it does not establish CUDA Graph or GPU lifecycle correctness.

## Static and repository checks

At import time, exact tracked diff comparison against the read-only production copy succeeded: SHA-256 `4e26a8087eb3c565e87d0ae7d74ea6056358260e1d01d30fd49c91b4066b687c`, 32 changed tracked files, 882 insertions, 100 deletions. Four additional untracked tests and the existing `profiles/2xT10/` material were copied separately and hash checked where applicable. The manifest hashes the exact imported snapshot before the later comment-only Apache file notices.

The final profile matrix and CMP50HX entry passed:

```text
$ bash tools/validate_profiles.sh
profile_validation_ok total=28
```

Gitleaks scanned both the working tree and full Git history using redacted JSON reports. The working-tree scan reported 12 heuristic hits; the full-history scan reported 99 hits across 24 commits. Sanitized hit metadata classified these as generic `key` identifiers/fields, token-count parameters, model identifiers, and documentation examples using a redacted bearer placeholder. The report's secret fields were `REDACTED`; no credential value was exposed or identified in the scan review. The scanner exits nonzero on these upstream false positives, so this is a reviewed-finding result, not a clean zero-finding scan. No production or GPU test was run during repository creation.

## Not verified

- Clean native build on CMP50HX, runtime library NCCL identity, and `nvcc` availability.
- GPU preemption/resume, full GDN lifecycle, CUDA Graph replay after slot reuse, and long-duration acceptance.
- Comparable PP/TG benchmarks or a speedup claim.
- Root cause of the 2026-11-10 host reboots and exact reboot-time source provenance.
- Production service health after publication work; this repository task must not restart or modify production B.
