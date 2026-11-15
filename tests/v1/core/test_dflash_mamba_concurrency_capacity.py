# SPDX-License-Identifier: Apache-2.0
"""DFlash align-mode Mamba pools must reserve for configured concurrency."""
from types import SimpleNamespace

import torch

from vllm.v1.kv_cache_interface import MambaSpec


def _config(max_num_seqs: int):
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        speculative_config=SimpleNamespace(method="dflash"),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_align_dflash_mamba_pool_budget_scales_with_max_num_seqs():
    spec = MambaSpec(
        block_size=16,
        shapes=((4,),),
        dtypes=(torch.uint8,),
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )

    single_request_bytes = spec.max_memory_usage_bytes(_config(1))
    two_request_bytes = spec.max_memory_usage_bytes(_config(2))

    assert single_request_bytes == 11 * spec.page_size_bytes
    # One full long-request resident/hash footprint plus one concurrent short
    # request with a partial-prefix CoW block: 11 + (1 + 7 + 1) pages.
    assert two_request_bytes == 20 * spec.page_size_bytes
