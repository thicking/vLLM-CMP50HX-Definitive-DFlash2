# Recovered Experimental B downstream test; see PATCHES.md.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression for one-shot DFlash prefix-cache block-table shifting."""
from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator


def test_cache_restore_shift_persists_across_scheduler_lifecycle():
    shift_once = getattr(
        dflash_speculator, "shift_restored_prefix_block_tables", None
    )

    block_size = 16
    block_table = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107]], dtype=torch.int32
    )
    cached = torch.tensor([32], dtype=torch.int32)
    idx = torch.tensor([0], dtype=torch.int32)
    seq_lens = torch.tensor([34], dtype=torch.int32)

    def cpu_shift(table, indices, cached_tokens, lengths, size, cp_size=1):
        span = size * cp_size
        for row, slot in enumerate(indices.tolist()):
            shift = int(cached_tokens[slot]) // span
            needed = (int(lengths[row]) + span - 1) // span
            remaining = min(table.shape[1] - shift, needed)
            if shift:
                table[row, :remaining] = table[row, shift : shift + remaining].clone()

    original_shift = dflash_speculator.shift_draft_block_tables
    dflash_speculator.shift_draft_block_tables = cpu_shift
    groups = SimpleNamespace(
        input_block_tables=[block_table],
        kernel_block_sizes=[block_size],
        cp_size=1,
    )

    def scheduler_step():
        if callable(shift_once):
            shift_once(groups, [0], idx, cached, seq_lens)
        else:
            # Reproduce committed HEAD: it clears num_cached_tokens after
            # shifting, so the persistent-base assertion fails on old source.
            for gid in [0]:
                dflash_speculator.shift_draft_block_tables(
                    groups.input_block_tables[gid],
                    idx,
                    cached,
                    seq_lens,
                    groups.kernel_block_sizes[gid],
                    groups.cp_size,
                )
            cached.index_fill_(0, idx.to(dtype=torch.long), 0)

    try:
        # Prefix restore on admission: restored tokens have no draft KV.
        scheduler_step()
        assert block_table[0, :3].tolist() == [102, 103, 104]
        assert cached.tolist() == [32]
        first_mapping = block_table[0, :3].tolist()

        # Scheduler rebuilds the block table from persistent IDs each step.
        block_table[0] = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107])
        seq_lens.fill_(18)
        scheduler_step()
        assert block_table[0, :2].tolist() == first_mapping[:2]
        assert cached.tolist() == [32]

        # Preemption/resume restores the cached-prefix count for the new admission.
        block_table[0] = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107])
        cached[0] = 32
        seq_lens.fill_(18)
        scheduler_step()
        assert block_table[0, :2].tolist() == first_mapping[:2]
        assert cached.tolist() == [32]

        # A new prefix-restored request reusing this slot supplies its own count.
        block_table[0] = torch.tensor([200, 201, 202, 203, 204, 205, 206, 207])
        cached[0] = 16
        seq_lens.fill_(18)
        scheduler_step()
        assert block_table[0, :2].tolist() == [201, 202]
        assert cached.tolist() == [16]
        # Logical token positions map to the physical blocks after exactly one shift.
        assert [int(block_table[0, i // block_size]) for i in range(18)] == [
            201
        ] * 16 + [202] * 2

        # Ordinary speculative decode without a prefix hit leaves the mapping intact.
        block_table[0] = torch.tensor([300, 301, 302, 303, 304, 305, 306, 307])
        cached[0] = 0
        seq_lens.fill_(18)
        unchanged = block_table.clone()
        scheduler_step()
        assert torch.equal(block_table, unchanged)

    finally:
        dflash_speculator.shift_draft_block_tables = original_shift


if __name__ == "__main__":
    test_cache_restore_shift_persists_across_scheduler_lifecycle()
