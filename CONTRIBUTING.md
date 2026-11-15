# Contributing

This project is an experimental downstream derivative of vLLM and
weicj/vLLM-2080Ti-Definitive. Preserve all upstream author, copyright,
license, SPDX, NOTICE, and acknowledgement material. Do not rewrite original
commit authorship or present upstream code as project-original work.

## Changes

1. Read the relevant upstream and project-specific instructions, code, and
   tests before editing.
2. For a defect, capture a reproducer and identify the failing invariant.
3. Add or update a regression test. It should fail on the prior behavior and
   pass with the fix; explain any intentionally updated contract.
4. Keep engine changes narrow. Do not hide a defect by disabling DFlash2,
   CUDA Graphs, prefix caching, or other acceleration.
5. Separate source observations, CPU tests, GPU tests, runtime acceptance, and
   production status. Do not report one as evidence for another.
6. Benchmark equivalent work and report model/checkpoint, hardware, runtime
   version, warmup, prompt/output lengths, cache state, concurrency, sampling,
   and the exact throughput metric.
7. Keep Custom AllReduce disabled in CMP50HX configurations. Do not modify
   production services without explicit owner approval.
8. Check profiles with `bash tools/validate_profiles.sh`, scripts with
   `bash -n`, changed Python files with `python3 -m py_compile`, and the final
   diff with `git diff --check`.
9. Exclude model weights, virtual environments, compiled artifacts, cache,
   logs, prompts, transcripts, telemetry, backups, credentials, and private
   infrastructure configuration. Review the complete Git history and objects
   before publication.

## Upstream updates

Upstream changes are fetched and reviewed manually. Keep separate remotes for
`upstream-2080ti` and `upstream-vllm`; never configure automatic merges into
`main`. Record the source ref and any conflicts resolved by the downstream
maintainer.
