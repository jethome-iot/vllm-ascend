import torch

from vllm_ascend._310p.ops.fla.gdn_310 import _zero_states_without_initial


def _reference(initial_state: torch.Tensor, has_initial_state: torch.Tensor) -> torch.Tensor:
    """The original boolean-index assignment, kept here as the source of truth."""
    out = initial_state.clone()
    out[~has_initial_state, ...] = 0
    return out


def test_zero_states_matches_assignment_4d():
    # Real GDN shape: [num_seqs, num_v_heads, v_dim, k_dim], mask [num_seqs], mixed True/False.
    torch.manual_seed(0)
    n, h, dv, dk = 5, 2, 12, 16
    initial_state = torch.randn(n, h, dv, dk, dtype=torch.float16)
    has_initial_state = torch.tensor([True, False, True, False, True])

    out = _zero_states_without_initial(initial_state, has_initial_state)

    assert out.shape == initial_state.shape
    assert out.dtype == initial_state.dtype
    torch.testing.assert_close(out, _reference(initial_state, has_initial_state))


def test_zero_states_zeroes_nan_inf_rows():
    # A masked (no-initial-state) row holding NaN/Inf must become exactly 0, not leak the
    # non-finite values. This is why torch.where is used instead of ``state * mask`` —
    # ``NaN * 0 == NaN`` and ``Inf * 0 == NaN`` would survive the multiply.
    initial_state = torch.randn(3, 1, 2, 2, dtype=torch.float32)
    initial_state[1, 0, 0, 0] = float("nan")
    initial_state[1, 0, 0, 1] = float("inf")
    initial_state[1, 0, 1, 0] = float("-inf")
    has_initial_state = torch.tensor([True, False, True])

    out = _zero_states_without_initial(initial_state, has_initial_state)

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[1]) == 0  # masked row fully zeroed
    torch.testing.assert_close(out[0], initial_state[0])  # kept rows untouched
    torch.testing.assert_close(out[2], initial_state[2])


def test_zero_states_all_true_and_all_false():
    # dim-agnostic reshape: also exercise a 3-D state.
    initial_state = torch.randn(4, 3, 5, dtype=torch.float16)

    all_true = torch.ones(4, dtype=torch.bool)
    torch.testing.assert_close(_zero_states_without_initial(initial_state, all_true), initial_state)

    all_false = torch.zeros(4, dtype=torch.bool)
    assert torch.count_nonzero(_zero_states_without_initial(initial_state, all_false)) == 0
