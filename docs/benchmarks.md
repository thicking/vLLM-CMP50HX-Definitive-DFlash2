# Prefix-cache and runtime measurements

All measurements below are tied to the listed host/profile and are observational. The A/B common table is one run per prompt size/profile, not a statistical study. Request bodies and generated user content are not published.

## GP-24.1: archived A vs B common, matched no-spec profile

Both engines used the same two-CMP50HX host, target weights and token-ID corpus. Matched profile: TP2, target KV FP8, context 200,000, `max_num_seqs=2`, max batched tokens 2,560, Mamba align, prefix caching ON, CUDA Graph capture size 4, Custom AllReduce OFF, speculative decoding OFF. Every cold/warm request requested one output token. Physical PP is locally computed prompt positions divided by request prefill duration; TTFT is client request start to the first emitted token. Cached counts shown are aligned hit positions for the matching exact repeat.

| Prompt | Cold Physical PP A / B (tok/s) | Cold TTFT A / B (s) | Cached on exact repeat | Warm TTFT A / B (s) |
|---:|---:|---:|---:|---:|
| 16,384 | 983.52 / 1,017.82 | 16.7049 / 16.1423 | 15,680 (95.70%) | 0.8641 / 0.7927 |
| 32,768 | 881.75 / 962.28 | 37.2310 / 34.1194 | 31,360 (95.70%) | 1.9628 / 1.6702 |
| 65,536 | 723.08 / 860.51 | 90.7472 / 76.2656 | 64,288 (98.10%) | 2.4306 / 1.8987 |
| 98,304 | 607.84 / 775.93 | 161.8531 / 126.8803 | 97,216 (98.89%) | 2.6722 / 1.9385 |

B common was faster in this single pass at all four lengths for cold Physical PP and cold/warm TTFT. The measurement supports this tested no-spec profile only. A and B had different software stacks; clean-build parity and a general performance advantage are not inferred.

### Stack and provenance

- Archived A: source HEAD `4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be` with dirty tracked changes; Python 3.11.16, Torch 2.11.0+cu128, Triton 3.6.0, FlashInfer 0.6.8.post1.
- B common: recovered B source base `2d63e2e322b7488776d1fb822f3556dbd7334116`; package metadata `0.2.3.dev0+g2d63e2e32.d20260928.cu130`, Torch 2.13.0+cu130, Triton 3.7.1.
- The A candidate required an isolated launcher correction to expose CUDA 12.8 `nvcc` and GCC 13 for FlashInfer JIT. This established the cause of that startup failure; it did not establish a universal dependency compatibility matrix.

## B common vs production B K=7

This is a profile comparison, not an isolated DFlash factor test. The production K=7 profile used context 221,184, batched tokens 4,096, DFlash2 K=7 and its own hybrid KV pool sizing; B common above used context 200,000, batched tokens 2,560 and speculative decoding OFF.

| Prompt | B common cold PP | Production K=7 cold PP | K=7 warm TTFT | K=7 cached tokens (hit rate) |
|---:|---:|---:|---:|---:|
| 16,384 | 1,017.82 | 996 | 3.3701 s | 13,184 (80.47%) |
| 32,768 | 962.28 | 945 | 3.6331 s | 29,664 (90.53%) |
| 65,536 | 860.51 | 851 | 4.2139 s | 62,624 (95.56%) |
| 98,304 | 775.93 | 770 | 4.6397 s | 95,584 (97.23%) |

The different context, scheduler batching, KV layout and speculative mode prevent attributing the difference to K=7 alone.

## KV pressure and decode anomaly

- **Matched 64K pressure run, A vs B common:** after three distinct 65,536-token prefixes, A's repeat had 0 cached and TTFT 92.6922 s; B common's repeat retained 64,288 tokens and TTFT was 1.9327 s. Startup capacities were 468,656 tokens (A) and 478,571 (B common). This is one matched pressure case, not a universal eviction comparison.
- **Production K=7 96K residency:** with three distinct 98,304-token prompt histories, a repeat of the first retained 51,088 cached tokens, recomputed 47,216 and had TTFT 70.1789 s. A separate archived A pressure run recomputed all 98,304 with TTFT 162.9029 s; because the profiles differed, this is not a matched A/B result.
- **Single decode probe:** one 4,096-prompt / 128-output no-spec probe measured client TG A 33.20 tok/s and B common 13.83 tok/s. Production B K=7 previously measured about 103.08 tok/s for the same request shape. The B-common anomaly was not repeated sufficiently; cause is UNKNOWN and no general speed claim follows.

## GP-25.2 Open WebUI cache observations

The pre-fix sequence in one OWUI chat had four prompts with zero cached tokens and TTFT 18.79–20.19 s. In one post-fix synthetic sequence, cache hits appeared on turns 2–4:

| Turn | Input tokens | Cached tokens | TTFT |
|---:|---:|---:|---:|
| 1 | 15,398 | 0 | 18.10 s |
| 2 | 16,096 | 13,184 | 3.15 s |
| 3 | 16,797 | 13,184 | 3.85 s |
| 4 | 17,537 | 14,832 | 2.88 s |

A later owner-observed sequence also reported cache hits on all four turns: input 15,431 / 16,404 / 17,877 / 18,794; cached 13,184 / 13,184 / 14,832 / 14,832; TTFT 5.90 / 3.36 / 3.21 / 5.68 s. The latter is preliminary real-use observation, not a controlled benchmark.

The earlier and later messages were not an exact replay of identical prompt token IDs. These figures show that caching worked in the tested post-fix sequences; they do not quantify a statistically controlled TTFT improvement or explain every historical miss.

## Production capacity and concurrency context

The GP-21.3 production allocation logged 247,398 scalar KV tokens, context 221,184, and 1.12x reported maximum concurrency. Its target full-attention group had 151 total / 150 usable 1,648-token blocks; the five draft pools were independent. The scalar is not a promise of two fully occupied maximum-length contexts. GP-21.3 separately verified overlapping work for the tested equal prompt pairs through 95,996 tokens and an asymmetric 199,990 + 16,000 pair; those tests are recorded in [TEST-REPORT.md](../TEST-REPORT.md).

## What these numbers do not show

- They do not establish a clean build/install from the public GitHub checkout.
- They do not establish a general upstream regression in Open WebUI 0.13.3.
- They do not establish that Prefix Cache is always resident under arbitrary hybrid KV pressure.
- They do not isolate DFlash K=7 performance from other profile changes.
- One-off A/B or decode results are not statistically significant.
