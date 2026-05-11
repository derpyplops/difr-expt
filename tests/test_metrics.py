"""Tests for metrics module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from difr_expt.metrics import (
    kl_div_ref_to_cand,
    logit_l2,
    post_gumbel_margin,
    top1_match,
    topk_overlap,
)


def test_top1_match_identical():
    torch.manual_seed(0)
    logits = torch.randn(8, 100)
    m = top1_match(logits, logits)
    assert m.all()


def test_top1_match_shifted():
    logits_a = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    logits_b = torch.tensor([[2.0, 1.0, 0.5], [0.0, 0.0, 1.0]])  # differ at row 0
    m = top1_match(logits_a, logits_b)
    assert m.tolist() == [False, True]


def test_topk_overlap_identical():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    assert torch.allclose(topk_overlap(logits, logits, k=5), torch.ones(4))


def test_topk_overlap_disjoint():
    # Two distributions with disjoint top-k.
    a = torch.zeros(1, 10)
    a[0, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    b = torch.zeros(1, 10)
    b[0, 5:] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    assert topk_overlap(a, b, k=5).item() == 0.0


def test_logit_l2_zero_for_identical():
    logits = torch.randn(3, 50)
    assert logit_l2(logits, logits).max() < 1e-6


def test_kl_zero_for_identical():
    logits = torch.randn(3, 50)
    assert kl_div_ref_to_cand(logits, logits).max() < 1e-6


def test_post_gumbel_margin_zero_when_logits_agree():
    """If ref and cand logits are equal, the verifier's preferred token equals
    the candidate's argmax under shared noise; margin should be ~0."""
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    noise = torch.empty(4, 100).exponential_().log().neg()  # Gumbel
    m = post_gumbel_margin(logits, logits, noise)
    assert m.abs().max() < 1e-6, m


def test_post_gumbel_margin_positive_when_disagree():
    """Force a strong disagreement at position 0: ref strongly prefers token 0,
    cand strongly prefers token 1, no noise. Margin should be positive."""
    ref = torch.tensor([[10.0, 0.0]])
    cand = torch.tensor([[0.0, 10.0]])
    noise = torch.zeros(1, 2)
    m = post_gumbel_margin(ref, cand, noise, temperature=1.0)
    assert m.item() > 0


if __name__ == "__main__":
    g = dict(globals())
    fails = []
    for name, fn in g.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                fails.append(name)
    print(f"failures: {fails}")
    sys.exit(1 if fails else 0)
