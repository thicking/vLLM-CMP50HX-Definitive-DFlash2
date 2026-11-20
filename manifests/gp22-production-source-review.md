# GP-22 production-source SHA-256 review

Read-only comparison against the deployed runtime source on 2026-11-20. Paths below are repository-relative; no host path or machine configuration is included.

| File | Candidate SHA-256 | Production SHA-256 | Result |
| --- | --- | --- | --- |
| `vllm/config/cache.py` | `14fd50bc9011e6b58065bd8fc9b9e54ebe54e73506e408cd46b267ed85aad4f3` | `14fd50bc9011e6b58065bd8fc9b9e54ebe54e73506e408cd46b267ed85aad4f3` | exact |
| `vllm/engine/arg_utils.py` | `bb742af30021562903bc062ce0552d7e3d66421793d9b233ad9360e2dd9dcc89` | `bb742af30021562903bc062ce0552d7e3d66421793d9b233ad9360e2dd9dcc89` | exact |
| `vllm/v1/core/gp12_diagnostics.py` | `f759e3d86edf1cd5fb6c28e64a30ddcaaa266e6e48d2c72c4e8cc82416024de5` | `f759e3d86edf1cd5fb6c28e64a30ddcaaa266e6e48d2c72c4e8cc82416024de5` | exact |
| `vllm/v1/core/kv_cache_manager.py` | `534986b4866c22a7d3d94e437a24d88214144de1538d775f5a41aa7314604a81` | `534986b4866c22a7d3d94e437a24d88214144de1538d775f5a41aa7314604a81` | exact |
| `vllm/v1/core/kv_cache_utils.py` | `5e35773489bc0f003d64959c660bb13c2ad1b046cbd273d73a01e74464440d80` | `5e35773489bc0f003d64959c660bb13c2ad1b046cbd273d73a01e74464440d80` | exact |
| `vllm/v1/core/sched/output.py` | `f6b943c4b2afde003fa424dd7fe2da58138e6038838108e67403ba2b0cc34503` | `513c57a3ff0f0cbee3d8a5010bcaf94d59bf6bf63184caa31294fe24f89c8180` | comment wording only |
| `vllm/v1/core/sched/scheduler.py` | `31906ec209545edd71f0d99eff617513201da58cdba7ef7b0837d7d8afc3f5e6` | `31906ec209545edd71f0d99eff617513201da58cdba7ef7b0837d7d8afc3f5e6` | exact |
| `vllm/v1/kv_cache_interface.py` | `433cb3ae2674441be44fdff6f4a5a22287c8a516b50981818e6f241345e1feda` | `433cb3ae2674441be44fdff6f4a5a22287c8a516b50981818e6f241345e1feda` | exact |
| `vllm/v1/metrics/loggers.py` | `b5a6b4bf6075fbd748607e57ce3f29d2afefb8b602b303988b15c3e7773254f1` | `1ff58bff31947e51194f2ac96c50628966a8a009dce05baa2421f356d4fc59ac` | docstring line wrapping only |
| `vllm/v1/metrics/stats.py` | `9ca7ae3ae7557e226bc34eb0b09964c2289753a980101436029fa302897b515c` | `49fbce6b187385100af712e882f5811f5d9fc1fe29c143ebe6fd9eec434a42c4` | downstream notice/comment wording only |

The three non-identical files were diffed directly. Their executable statements and metric behavior are unchanged; differences are comments, documentation line wrapping, and the downstream modified-file notice. Seven of the ten changed engine files are byte-identical. This is source comparison only; it is not a fresh runtime test.
