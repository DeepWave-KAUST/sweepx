"""Resuming must restore Adam's moments, or say so loudly.

A resume that reloads only the network weights is not equivalent to a
continuous run -- Adam's first/second moments set the step scale, and without
them the first dozen steps move by a completely different amount. The failure
is silent and looks like a physics problem, so a missing state file raises
here instead of leaving a zero state behind.
"""
from types import SimpleNamespace

import pytest
import torch

from sweep_tasks.schemas import ReparamSpec
from sweep_tasks.tasks.fwi_freqsel import FreqselRunnerMixin

restore = FreqselRunnerMixin._restore_optimizer_state


def _spec(**rp):
    return SimpleNamespace(reparam=SimpleNamespace(**rp))


def _opt(seed=0):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(4))
    o = torch.optim.Adam([p], lr=1e-3)
    return p, o


def _stepped(n=5, seed=0):
    """An optimizer with real moments in it, not a fresh one."""
    p, o = _opt(seed)
    for _ in range(n):
        o.zero_grad(); (p ** 2).sum().backward(); o.step()
    return p, o


def test_unset_is_a_no_op():
    p, o = _opt()
    before = o.state_dict()
    restore(_spec(init_optimizer_from=None), o, 0)
    assert o.state_dict() == before
    restore(SimpleNamespace(reparam=None), o, 0)


def test_auto_without_init_from_is_rejected():
    _, o = _opt()
    with pytest.raises(ValueError, match="needs reparam.init_from"):
        restore(_spec(init_optimizer_from="auto", init_from=None), o, 0)


def test_missing_file_raises_instead_of_a_silent_zero_state(tmp_path):
    """The whole point of the feature."""
    _, o = _opt()
    with pytest.raises(FileNotFoundError, match="not equivalent to a continuous run"):
        restore(_spec(init_optimizer_from=str(tmp_path / "nope.pt")), o, 0)


def test_auto_derives_optim_iter_from_the_net_name(tmp_path):
    _, src = _stepped()
    torch.save({"optimizer": src.state_dict()}, tmp_path / "optim_iter0030.pt")
    _, dst = _opt(seed=1)
    restore(_spec(init_optimizer_from="auto",
                  init_from=str(tmp_path / "reparam_net_iter0030.pt")), dst, 0)
    assert dst.state_dict()["state"][0]["step"] == src.state_dict()["state"][0]["step"]


def test_auto_falls_back_to_optim_pt_beside_the_net(tmp_path):
    _, src = _stepped(n=7)
    torch.save({"optimizer": src.state_dict()}, tmp_path / "optim.pt")
    _, dst = _opt(seed=1)
    restore(_spec(init_optimizer_from="auto",
                  init_from=str(tmp_path / "reparam_net.pt")), dst, 0)
    assert dst.state_dict()["state"][0]["step"] == src.state_dict()["state"][0]["step"]


def test_moments_really_come_back(tmp_path):
    """Not just 'it loaded': the exp_avg tensors must match."""
    _, src = _stepped(n=9)
    torch.save({"optimizer": src.state_dict()}, tmp_path / "o.pt")
    _, dst = _opt(seed=2)
    restore(_spec(init_optimizer_from=str(tmp_path / "o.pt")), dst, 0)
    a, b = src.state_dict()["state"][0], dst.state_dict()["state"][0]
    assert torch.equal(a["exp_avg"], b["exp_avg"])
    assert torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])


def test_a_bare_state_dict_is_accepted(tmp_path):
    """Older snapshots were saved without the {'optimizer': ...} wrapper."""
    _, src = _stepped(n=3)
    torch.save(src.state_dict(), tmp_path / "bare.pt")
    _, dst = _opt(seed=3)
    restore(_spec(init_optimizer_from=str(tmp_path / "bare.pt")), dst, 0)
    assert dst.state_dict()["state"][0]["step"] == src.state_dict()["state"][0]["step"]


def test_schema_defaults():
    r = ReparamSpec()
    assert r.init_optimizer_from is None
    assert r.save_optimizer is True, "moments must be saved with the weights"
