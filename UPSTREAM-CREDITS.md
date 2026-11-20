## 🙏 Credits And Upstream Projects

This repository is a hardware-focused fork of
[vLLM](https://github.com/vllm-project/vllm), licensed under Apache-2.0. It
keeps the upstream project structure and adds local SM75 runtime patches,
launch profiles, and dual-2080-Ti validation notes.

Acceleration components used or integrated by this runtime include:

- [vLLM](https://github.com/vllm-project/vllm): base inference engine and
  serving stack.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): attention,
  sampling, and quantized kernel paths used by vLLM.
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA): upstream Gated
  DeltaNet / Qwen hybrid linear-attention implementation.
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75):
  SM70/SM75 adaptation used by the validated Qwen prefill route.
- TurboQuant, Marlin, CUTLASS, Triton, and related vLLM kernels.

Upstream updates are re-evaluated within the SM75-specific scope of this fork.

This repository is an independently maintained derivative of
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
and upstream vLLM. GP-20, GP-21.1, and GP-21.3 downstream changes are listed
with their exact scope in `PATCHES.md`; original authorship, notices, and
Apache-2.0 license terms are retained. A minimal GP-21.1 scheduler proposal
and issue draft are included locally under `upstream-proposals/` and have not
been submitted upstream.
