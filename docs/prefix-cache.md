# Prefix Cache: operational guide for the CMP50HX DFlash2 route

This note describes the observed vLLM prefix-cache behavior for the tested two-CMP50HX runtime. For the incident timeline and measured cases, see [Lessons learned (Russian)](lessons-learned.ru.md) and [benchmarks](benchmarks.md).

## What the cache can reuse

Prefix caching reuses complete matching prompt blocks from the beginning of a sequence. It does not search for a matching suffix after an earlier mismatch. In the production DFlash2 K=7 route, the observed aligned cache block was 1,648 tokens; in the separate no-spec common A/B profile it was 1,568 tokens. Do not assume one block size applies to every hybrid cache layout.

If the first changing token is before the first complete block boundary, two requests may share most of their text and still have zero reusable full blocks. Changes later in a prompt can preserve the earlier shared prefix; changes near the beginning invalidate that later text as a prefix candidate.

## Keep the serialized beginning stable

- Keep system instructions deterministic across requests. Avoid injecting current time or request-specific values before the stable shared prefix.
- Query time or other dynamic state only when the task needs it.
- Keep tool schema content and order stable when the available tool set is unchanged. A real schema addition/removal is a legitimate prompt change.
- Preserve model-specific system instructions; do not replace distinct model presets with one generic prompt just to increase cache hits.
- Treat user messages, tool results, reasoning mode, chat template, tool schemas and multimodal payload as potential prompt changes; measure token IDs rather than guessing from visible text.

GP-25.2 removed time placeholders from the beginning of four configured system prompts and three Skills while preserving their other instructions. It was applied through Open WebUI configuration records, not through an inference-engine change. Live OWUI sequences then reported cache hits on subsequent requests. Those results are observational sequences, not a strict replay benchmark.

## Miss diagnosis

For a single request, record the request-level `prompt_tokens` and `cached_tokens` reported by the serving API. Compare sequential requests using token IDs or privacy-preserving block fingerprints. Record the common-prefix length and first differing block. Do not log raw prompts, tool results, images, API keys, cookies or authorization headers.

Then separate two failure classes:

1. **Payload mismatch:** the current token prefix differs, often due to dynamic prompt content, tool schema changes/order, or message serialization.
2. **Residency miss:** the prefix matched earlier but its blocks were evicted under the active KV working set.

Request usage is the source for request-level cache reuse. An aggregate `prefix_cache_hits_total` rate is not a per-request cache hit and should not be presented as physical prefill throughput.

## Hybrid KV capacity is group-specific

The production profile uses target FP8 KV, draft FP16 KV and multiple independent Mamba, attention and DFlash draft groups. The logged scalar KV capacity is a useful startup summary, not a promise that every hybrid group can hold that many tokens for every workload. The production full-attention group had 150 usable blocks of 1,648 tokens each in the recorded GP-21.3 allocation. DFlash draft pools and state reservations impose separate admission constraints.

Under pressure, a repeated 98,304-token K7 request retained 51,088 cached tokens and recomputed 47,216, with TTFT around 70.18 seconds. A different low-pressure exact-repeat test retained 95,584. These values describe residency under distinct request histories; they do not establish a cache hashing defect.

## Metrics contracts

- `cached_tokens`: prompt positions served from prefix cache for a specific request, when the API reports it.
- `vllm:prefill_computed_tokens_total`: completed local prefill positions. Prefix-cache hits, external KV, decode and speculative verification are excluded; recomputation after preemption counts as actual execution.
- `prefix_cache_hits_total`: aggregate cache-hit positions observed by the metrics pipeline. A rate derived from this counter is not GPU Physical PP.

Use the request-level value to answer “did this prompt reuse blocks?” and the live physical-prefill metric to answer “how many prompt positions were computed?”

## Configuration and evidence boundaries

The tested production route was TP2, context 221,184, `max_num_seqs=2`, DFlash2 K=7, CUDA Graphs, prefix caching, Mamba align and Custom AllReduce OFF. The reported maximum context is per request; shared cache capacity and scheduler admission determine which concurrent workload fits. Do not reduce context, change draft settings or disable cache based only on a scalar capacity or a single miss.

The public repository has GPU runtime-validation evidence for the recovered Experimental B tree. A clean native build/install directly from the public checkout remains NOT VERIFIED. The full test and release evidence is in [TEST-REPORT.md](../TEST-REPORT.md).
