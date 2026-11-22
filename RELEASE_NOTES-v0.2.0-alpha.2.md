# v0.2.0-alpha.2 — Operational findings and benchmark documentation

This is a documentation-only prerelease for the independently maintained vLLM-CMP50HX-Definitive-DFlash2 derivative.

## Included

- A Russian technical lessons-learned record for the GP-24/GP-25 Prefix Cache investigation.
- A Prefix Cache guide describing stable prompt prefixes, block alignment, request-level evidence, and hybrid KV residency.
- A benchmark record with the measured A/B profiles, production K=7 observations, and explicit evidence limitations.

## Runtime and validation

No engine source, profile, launcher, service, dependency, model, or production runtime setting changes in this release. It adds no new GPU measurements. The existing runtime validation and its limitations remain in `TEST-REPORT.md`; a clean native build directly from the public checkout remains NOT VERIFIED.

## Documents

- [Lessons learned (Russian)](docs/lessons-learned.ru.md)
- [Prefix Cache guide](docs/prefix-cache.md)
- [Measurements and profiles](docs/benchmarks.md)
- [Validation record](TEST-REPORT.md)
