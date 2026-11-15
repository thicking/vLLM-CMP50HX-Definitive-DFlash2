# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.dflash2_sm75 import (
    Sm75DFlash2MLP,
    Sm75DFlash2ResidualNorm,
    encode_dflash2_mlp_sm75,
    restore_dflash2_mlp_sm75,
    should_enable_dflash2_sm75,
    synchronize_dflash2_mlp_scales,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3Model,
    _grouped_conv,
    _score_edges,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_grouped_conv_preserves_a_wide_sm75_residual_boundary():
    hidden = torch.full((1, 2), 40000.0, dtype=torch.float16)
    delta = torch.zeros((1, 1, 1), dtype=torch.float16)
    base = torch.full((1, 2), 2.0, dtype=torch.float16)

    actual = _grouped_conv(
        hidden,
        delta,
        base,
        block_size=1,
        num_groups=1,
        group_size=2,
        taps=1,
        output_dtype=torch.float32,
    )

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, torch.full((1, 2), 80000.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_grouped_conv_cuda_matches_sm75_decode_shape(output_dtype):
    torch.manual_seed(20260916)
    rows, block_size, taps = 8, 8, 2
    num_groups, group_size = 320, 16
    hidden = torch.randn(
        rows,
        num_groups * group_size,
        device="cuda",
        dtype=torch.float16,
    )
    delta = torch.randn(
        rows,
        taps,
        num_groups,
        device="cuda",
        dtype=torch.float16,
    )
    base = torch.randn(
        taps,
        num_groups * group_size,
        device="cuda",
        dtype=torch.float16,
    )

    actual = _grouped_conv(
        hidden,
        delta,
        base,
        block_size,
        num_groups,
        group_size,
        taps,
        output_dtype=output_dtype,
    )

    hidden_blocks = hidden.float().view(1, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(1, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]
    expected = expected.flatten(0, 1).flatten(-2)
    if output_dtype is None:
        expected = expected.to(hidden.dtype)

    assert actual.dtype is (output_dtype or hidden.dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def _bf16_boundary(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def test_sm75_dflash2_mlp_codec_preserves_bf16_boundaries_and_range():
    gate_up = torch.tensor(
        [[1.75, -2.5, 3.25, -4.0, 1024.0, -512.0, 256.0, -128.0]],
        dtype=torch.float16,
    )

    down_projection_l2_bound = 128.0
    accumulation_limit = 16376.0
    payload, scales = encode_dflash2_mlp_sm75(
        gate_up,
        down_projection_l2_bound=down_projection_l2_bound,
        accumulation_limit=accumulation_limit,
    )
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = _bf16_boundary(gate * 32.0)
    up = _bf16_boundary(up * 32.0)
    expected = _bf16_boundary(_bf16_boundary(gate * torch.sigmoid(gate)) * up)
    expected_scale = torch.exp2(
        torch.ceil(
            torch.log2(
                torch.maximum(
                    torch.maximum(
                        expected.abs().amax(dim=-1) / 32752.0,
                        torch.linalg.vector_norm(expected, dim=-1)
                        * down_projection_l2_bound
                        / accumulation_limit,
                    ),
                    torch.ones(1),
                )
            )
        )
    )

    torch.testing.assert_close(payload.float() * scales[:, None], expected)
    torch.testing.assert_close(scales, expected_scale)
    assert bool((payload.abs() <= 32752.0).all())
    assert bool((torch.log2(scales) == torch.round(torch.log2(scales))).all())


def test_sm75_dflash2_mlp_codec_scales_for_down_projection_l2_bound():
    gate_up = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
    bound = 64.0
    limit = 16.0

    _payload, scales = encode_dflash2_mlp_sm75(
        gate_up,
        down_projection_l2_bound=bound,
        accumulation_limit=limit,
    )
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = _bf16_boundary(gate * 32.0)
    up = _bf16_boundary(up * 32.0)
    product = _bf16_boundary(_bf16_boundary(gate * torch.sigmoid(gate)) * up)

    # The unrounded payload's Cauchy-Schwarz upper bound fits the local budget.
    assert bool(
        (torch.linalg.vector_norm(product, dim=-1) * bound / scales <= limit).all()
    )


def test_sm75_dflash2_quantized_bound_uses_int8_metadata_without_dequantizing():
    class QuantMethod:
        quant_config = SimpleNamespace(weight_bits=8, is_sym=True)

        def apply(self, _layer, value):
            return value

    down_proj = SimpleNamespace(
        qweight=torch.ones((2, 3), dtype=torch.int32),
        scales=torch.tensor([[0.125, 0.25], [0.5, 0.125]]),
        input_size_per_partition=8,
        tp_size=2,
        quant_method=QuantMethod(),
    )
    mlp = SimpleNamespace(gate_up_proj=torch.nn.Identity(), down_proj=down_proj)

    assert Sm75DFlash2MLP.supports(mlp)
    wrapper = Sm75DFlash2MLP(mlp)
    wrapper.initialize_down_projection_bound()

    # Global max scale bounds every row while qweight remains packed.
    assert wrapper._down_projection_l2_bound == pytest.approx(
        0.5 * 128 * 8**0.5
    )
    assert wrapper._accumulation_limit == pytest.approx(32752.0 / 2)


def test_sm75_dflash2_quantized_mlp_restores_the_scale_contract_once():
    class QuantMethod:
        def __init__(self, apply):
            self.quant_config = SimpleNamespace(weight_bits=8, is_sym=True)
            self.apply = apply

    class FakeQuantLinear(torch.nn.Module):
        def __init__(self, apply):
            super().__init__()
            self.quant_method = QuantMethod(apply)
            self.last_input = None
            self.qweight = torch.ones((1, 1), dtype=torch.int32)
            self.scales = torch.tensor([[0.25]])
            self.input_size_per_partition = 2
            self.tp_size = 1

        def forward(self, value):
            self.last_input = value.detach().clone()
            return self.quant_method.apply(self, value), None

    gate_up_value = torch.tensor(
        [[0.25, -0.5, 1.0, 2.0], [-0.75, 0.5, 1.5, -1.0]],
        dtype=torch.float16,
    )
    gate_up = FakeQuantLinear(lambda _layer, _x: gate_up_value)
    down_weight = torch.tensor(
        [[0.5, -0.25], [0.125, 0.75]], dtype=torch.float16
    )
    down = FakeQuantLinear(
        lambda _layer, value: torch.nn.functional.linear(value, down_weight)
    )
    mlp = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    wrapper = Sm75DFlash2MLP(mlp)
    wrapper.initialize_down_projection_bound()

    x = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.float16)
    actual = wrapper(x)

    torch.testing.assert_close(gate_up.last_input, x / 32.0)
    expected_payload, expected_scales = encode_dflash2_mlp_sm75(
        gate_up_value,
        down_projection_l2_bound=wrapper._down_projection_l2_bound,
        accumulation_limit=wrapper._accumulation_limit,
    )
    torch.testing.assert_close(down.last_input, expected_payload)
    expected_projected = torch.nn.functional.linear(expected_payload, down_weight)
    expected = expected_projected.float() * (expected_scales[:, None] / 256.0)
    torch.testing.assert_close(actual, expected.to(actual.dtype))


def test_sm75_dflash2_restore_matches_row_scale_contract():
    output = torch.tensor([[4.0, -2.0], [0.5, -1.0]], dtype=torch.float16)
    scales = torch.tensor([1.0, 8.0], dtype=torch.float32)

    actual = restore_dflash2_mlp_sm75(output.clone(), scales)
    expected = output.float() * (scales[:, None] / 256.0)

    torch.testing.assert_close(actual, expected.to(torch.float16))


def test_sm75_dflash2_tp_scales_share_one_reduction_domain(monkeypatch):
    payload = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float16)
    scales = torch.tensor([2.0, 8.0], dtype=torch.float32)

    monkeypatch.setattr(
        "vllm.model_executor.models.dflash2_sm75.tensor_model_parallel_all_gather",
        lambda value, dim=0: torch.cat((value, value * 2), dim=dim),
    )
    actual_payload, actual_scales = synchronize_dflash2_mlp_scales(
        payload, scales, tp_size=2
    )

    torch.testing.assert_close(actual_scales, torch.tensor([4.0, 16.0]))
    torch.testing.assert_close(actual_payload, payload * 0.5)


def test_sm75_dflash2_rmsnorm_uses_fp32_residual_and_bf16_boundaries():
    norm = Sm75DFlash2ResidualNorm(hidden_size=4, eps=1e-6, dtype=torch.float16)
    norm.weight.data.copy_(torch.tensor([1.0, 0.5, -1.0, 2.0], dtype=torch.float16))
    x = torch.tensor([[0.25, -1.5, 2.0, -0.75]], dtype=torch.float16)
    residual = torch.tensor([[400.0, -200.0, 80.0, -40.0]], dtype=torch.float32)

    actual, actual_residual = norm(x, residual)
    expected_residual = _bf16_boundary(x.float() * 256.0 + residual)
    expected = _bf16_boundary(
        expected_residual
        * torch.rsqrt(expected_residual.square().mean(dim=-1, keepdim=True) + 1e-6)
        * _bf16_boundary(norm.weight.float())
    ).to(torch.float16)

    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(actual, expected)


def test_sm75_dflash2_rmsnorm_returns_fp16_from_a_wide_residual_input():
    norm = Sm75DFlash2ResidualNorm(hidden_size=2, eps=1e-6, dtype=torch.float16)
    x = torch.tensor([[100000.0, -50000.0]], dtype=torch.float32)
    residual = torch.tensor([[1000.0, -2000.0]], dtype=torch.float32)

    actual, state = norm(x, residual)
    expected_state = _bf16_boundary(x * 256.0 + residual)
    expected = _bf16_boundary(
        expected_state
        * torch.rsqrt(expected_state.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).to(torch.float16)

    assert actual.dtype is torch.float16
    assert state.dtype is torch.float32
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(actual, expected)


def test_dflash_context_kv_projection_uses_quantized_qkv_modules():
    class QuantizedProjection:
        def __init__(self, kv_values: torch.Tensor):
            self.kv_values = kv_values

        def __call__(self, hidden_states: torch.Tensor):
            query = hidden_states.new_zeros((hidden_states.shape[0], 2))
            kv = self.kv_values.expand(hidden_states.shape[0], -1)
            return torch.cat((query, kv), dim=-1), None

    values = [
        torch.tensor([[10.0, 11.0, 20.0, 21.0], [12.0, 13.0, 22.0, 23.0]]),
        torch.tensor([[30.0, 31.0, 40.0, 41.0], [32.0, 33.0, 42.0, 43.0]]),
    ]
    layers_attn = [
        SimpleNamespace(
            qkv_proj=QuantizedProjection(kv),
            q_size=2,
            k_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2))),
        )
        for kv in values
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(4))),
        _normalize_context_states=lambda states: states,
    )
    DFlashQwen3Model._build_context_kv_buffers(model, layers_attn, has_bias=False)

    assert model._fused_kv_weight is None
    context = torch.zeros(2, 4)
    actual_k, actual_v = DFlashQwen3Model._project_context_kv(
        model, context, num_ctx=2, num_layers=2, num_kv_heads=1, head_dim=2
    )
    expected_k = torch.stack([value[:, :2].view(2, 1, 2) for value in values])
    expected_v = torch.stack([value[:, 2:].view(2, 1, 2) for value in values])
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)


def test_dflash_context_kv_projection_keeps_dense_fused_gemm():
    weights = [
        torch.nn.Parameter(torch.arange(24, dtype=torch.float32).view(6, 4)),
        torch.nn.Parameter(torch.arange(24, 48, dtype=torch.float32).view(6, 4)),
    ]
    layers_attn = [
        SimpleNamespace(
            qkv_proj=SimpleNamespace(weight=weight, bias=None),
            q_size=2,
            k_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2))),
        )
        for weight in weights
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(4))),
        _normalize_context_states=lambda states: states,
    )
    DFlashQwen3Model._build_context_kv_buffers(model, layers_attn, has_bias=False)

    assert model._fused_kv_weight is not None
    context = torch.arange(8, dtype=torch.float32).view(2, 4)
    actual_k, actual_v = DFlashQwen3Model._project_context_kv(
        model, context, num_ctx=2, num_layers=2, num_kv_heads=1, head_dim=2
    )
    expected_flat = torch.nn.functional.linear(
        context, torch.cat([weight[2:] for weight in weights])
    )
    expected = expected_flat.view(2, 2, 2, 1, 2).permute(2, 1, 0, 3, 4)
    torch.testing.assert_close(actual_k, expected[0].contiguous())
    torch.testing.assert_close(actual_v, expected[1].contiguous())


def test_sm75_dflash2_uses_the_context_norm_boundary():
    class ContextNorm:
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return value + 3

    model = SimpleNamespace(
        layers=[SimpleNamespace(use_sm75_transport=True)],
        hidden_norm=ContextNorm(),
    )
    states = torch.ones(2, 4)

    actual = DFlash2Qwen3Model._normalize_context_states(model, states)

    torch.testing.assert_close(actual, torch.full_like(states, 4))


def test_sm75_codec_requires_the_exact_sm75_bf16_fp16_tuple(monkeypatch):
    config = SimpleNamespace(torch_dtype="bfloat16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))

    assert should_enable_dflash2_sm75(config, torch.float16)
    assert not should_enable_dflash2_sm75(config, torch.float32)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert not should_enable_dflash2_sm75(config, torch.float16)


def test_sm75_codec_can_be_disabled_for_baseline_comparison(monkeypatch):
    config = SimpleNamespace(torch_dtype="bfloat16")
    monkeypatch.setenv("VLLM_DFLASH2_SM75_TRANSPORT", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))

    assert not should_enable_dflash2_sm75(config, torch.float16)


def test_sm75_codec_accepts_a_dtype_only_checkpoint_config(monkeypatch):
    config = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))

    assert should_enable_dflash2_sm75(config, torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm75_cuda_codec_matches_reference_at_dflash2_verifier_shape():
    """Exercise the actual SM75 kernels at the B=8 DFlash2 verifier shape."""
    torch.manual_seed(20260914)
    device = torch.device("cuda")
    rows, hidden_size, intermediate_size = 8, 5120, 17408
    down_projection_l2_bound = 128.0
    accumulation_limit = 16376.0
    gate_up = torch.randn(
        rows, 2 * intermediate_size, device=device, dtype=torch.float16
    ) * 1024

    expected_payload, expected_scales = encode_dflash2_mlp_sm75(
        gate_up.cpu(),
        down_projection_l2_bound=down_projection_l2_bound,
        accumulation_limit=accumulation_limit,
    )
    payload, scales = encode_dflash2_mlp_sm75(
        gate_up,
        down_projection_l2_bound=down_projection_l2_bound,
        accumulation_limit=accumulation_limit,
    )
    torch.testing.assert_close(scales.cpu(), expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(payload.cpu(), expected_payload, rtol=0, atol=0)

    projected = torch.randn(
        rows, hidden_size, device=device, dtype=torch.float16
    )
    projected_before = projected.clone()
    restored = restore_dflash2_mlp_sm75(projected, scales)
    expected_restored = restore_dflash2_mlp_sm75(
        projected.cpu(), expected_scales
    )
    assert restored.data_ptr() != projected.data_ptr()
    torch.testing.assert_close(projected, projected_before, rtol=0, atol=0)
    torch.testing.assert_close(restored.cpu(), expected_restored, rtol=0, atol=0)

    norm = Sm75DFlash2ResidualNorm(hidden_size, eps=1e-6, dtype=torch.float16).to(
        device
    )
    norm.weight.data.copy_(
        torch.randn(hidden_size, device=device, dtype=torch.float16)
    )
    x = torch.randn(rows, hidden_size, device=device, dtype=torch.float16)
    residual = torch.randn(rows, hidden_size, device=device, dtype=torch.float32) * 256
    actual, actual_state = norm(x, residual)
    expected_state = _bf16_boundary(x.cpu().float() * 256.0 + residual.cpu())
    expected = _bf16_boundary(
        expected_state
        * torch.rsqrt(expected_state.square().mean(dim=-1, keepdim=True) + 1e-6)
        * _bf16_boundary(norm.weight.cpu().float())
    ).to(torch.float16)
    torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    wide_x = x.float() * 100000.0
    wide_output, wide_state = norm(wide_x, residual)
    wide_expected_state = _bf16_boundary(
        wide_x.cpu() * 256.0 + residual.cpu()
    )
    wide_expected = _bf16_boundary(
        wide_expected_state
        * torch.rsqrt(
            wide_expected_state.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        * _bf16_boundary(norm.weight.cpu().float())
    ).to(torch.float16)
    assert wide_output.dtype is torch.float16
    assert wide_state.dtype is torch.float32
    torch.testing.assert_close(wide_state.cpu(), wide_expected_state, rtol=0, atol=0)
    torch.testing.assert_close(wide_output.cpu(), wide_expected, rtol=0, atol=0)


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skip_global_cleanup
def test_dflash2_model_decoder_layer_cls(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 3,
            "conv_group_size": 2,
            "use_aux_hidden_state": False,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model = DFlash2Qwen3Model(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], DFlash2Qwen3DecoderLayer)
