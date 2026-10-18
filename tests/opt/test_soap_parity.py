"""Bit-for-bit parity of sweepx.opt.optim.SOAP with the official implementation.

The reference is a verbatim copy of https://github.com/nikhilvyas/SOAP
(commit a1e5535) in tests/_reference/soap_official.py. Both optimizers get the
same parameters and the same gradients for enough steps to refresh the basis
several times; parameters and every piece of optimizer state must be equal
(``torch.equal``), not merely close.
"""
import hashlib
import importlib.util
import pathlib

import pytest
import torch

from sweepx.opt.optim import SOAP

_REF_PATH = pathlib.Path(__file__).parent / "_reference" / "soap_official.py"
_REF_SHA256 = "9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b"


def _load_reference():
    spec = importlib.util.spec_from_file_location("soap_official", _REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SOAP


SOAPRef = _load_reference()


def test_reference_is_verbatim():
    body = _REF_PATH.read_bytes().split(b"# ---- verbatim below ----\n", 1)[1]
    assert hashlib.sha256(body).hexdigest() == _REF_SHA256


def _trajectory(make_opt, inits, grads, groups, gamma=None, skip=()):
    """Run one optimizer on fresh copies of ``inits`` with the given gradients."""
    params = [torch.nn.Parameter(x.clone()) for x in inits]
    opt = make_opt([dict(g, params=[params[i] for i in g["params"]]) for g in groups])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma) if gamma else None
    for t, gs in enumerate(grads):
        for i, (p, g) in enumerate(zip(params, gs)):
            p.grad = None if (t, i) in skip else g.clone()
        opt.step()
        if sched is not None:
            sched.step()
    return params, opt


def _assert_identical(p_ref, o_ref, p_ours, o_ours):
    for a, b in zip(p_ref, p_ours):
        assert torch.equal(a.detach(), b.detach())
    for a, b in zip(p_ref, p_ours):
        s_ref, s_ours = o_ref.state[a], o_ours.state[b]
        assert s_ref["step"] == s_ours["step"]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(s_ref[key], s_ours[key]), key
        for key in ("GG", "Q"):
            assert len(s_ref[key]) == len(s_ours[key])
            for x, y in zip(s_ref[key], s_ours[key]):
                if isinstance(x, list):                # official marks "no factor" with []
                    assert len(x) == 0 and y is None, key
                else:
                    assert torch.equal(x, y), key


# (id, shapes, per-group kwargs with param indices, extra kwargs for both, steps, grad scale, scheduler gamma, skipped (step, param))
CASES = [
    ("matrix-defaults", [(6, 5)], [dict(params=[0])], dict(), 25, 1.0, None, ()),
    ("matrix-f1-betas", [(6, 5)], [dict(params=[0], betas=(0.9, 0.999))], dict(precondition_frequency=1), 25, 1.0, None, ()),
    ("matrix-tiny-grads", [(8, 3)], [dict(params=[0], betas=(0.9, 0.999))], dict(precondition_frequency=1), 25, 1e-6, None, ()),
    ("matrix-wd-shampoo-beta", [(6, 5)], [dict(params=[0], weight_decay=0.01, shampoo_beta=0.9)], dict(precondition_frequency=3), 25, 1.0, None, ()),
    ("long-axes-identity", [(20, 1), (3, 20)], [dict(params=[0, 1])], dict(max_precond_dim=8, precondition_frequency=4), 25, 1.0, None, ()),
    ("vector", [(7,)], [dict(params=[0])], dict(precondition_frequency=2), 15, 1.0, None, ()),
    ("vector-precondition-1d", [(7,)], [dict(params=[0])], dict(precondition_1d=True, precondition_frequency=2), 15, 1.0, None, ()),
    ("tensor-3d", [(4, 3, 5)], [dict(params=[0])], dict(precondition_frequency=2), 15, 1.0, None, ()),
    ("tensor-3d-merge", [(4, 3, 5)], [dict(params=[0])], dict(merge_dims=True, max_precond_dim=12, precondition_frequency=2), 15, 1.0, None, ()),
    ("tensor-4d-merge", [(3, 2, 2, 2)], [dict(params=[0])], dict(merge_dims=True, max_precond_dim=8, precondition_frequency=2), 15, 1.0, None, ()),
    ("normalize-no-bias-corr", [(5, 4)], [dict(params=[0], normalize_grads=True, correct_bias=False)], dict(precondition_frequency=3), 15, 1.0, None, ()),
    ("groups-scheduler-skips", [(6, 5), (4,), (3, 4)],
     [dict(params=[0, 2], lr=1e-2, betas=(0.9, 0.999)), dict(params=[1], lr=3e-3)],
     dict(precondition_frequency=2), 20, 1.0, 0.9, ((0, 1), (3, 1), (4, 0), (5, 2))),
]


def _check(case, device):
    _, shapes, groups, extra, steps, scale, gamma, skip = case
    gen = torch.Generator().manual_seed(0)
    inits = [torch.randn(*s, generator=gen).to(device) for s in shapes]
    grads = [[(scale * torch.randn(*s, generator=gen)).to(device) for s in shapes] for _ in range(steps)]
    wd = 0.0                                           # sweepx.opt default; official default is 0.01
    ours_kw = {("update_freq" if k == "precondition_frequency" else k): v for k, v in extra.items()}
    pa, oa = _trajectory(lambda g: SOAPRef(g, weight_decay=wd, **extra), inits, grads, groups, gamma, skip)
    pb, ob = _trajectory(lambda g: SOAP(g, weight_decay=wd, **ours_kw), inits, grads, groups, gamma, skip)
    _assert_identical(pa, oa, pb, ob)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_soap_matches_official_cpu(case):
    _check(case, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("case", CASES[:5], ids=[c[0] for c in CASES[:5]])
def test_soap_matches_official_cuda(case):
    _check(case, "cuda")
