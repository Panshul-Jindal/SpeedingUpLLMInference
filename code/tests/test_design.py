from itertools import combinations

import numpy as np
import pytest

from harness import optimizations as opt
from harness.config import load_settings
from harness.design import build_cells, full_factorial, half_fraction, model_matrix, smoke_configs
from harness.regimes import all_points


@pytest.mark.parametrize("n_gpus,n", [(1, 6), (4, 7)])
def test_counts(n_gpus, n):
    act = opt.active_numbers(n_gpus)
    assert len(act) == n
    assert len(full_factorial(act)) == 2 ** n
    assert len(half_fraction(act)) == 2 ** (n - 1)


@pytest.mark.parametrize("n_gpus", [1, 4])
def test_half_fraction_even_and_has_default(n_gpus):
    act = opt.active_numbers(n_gpus)
    hf = half_fraction(act)
    assert all(bin(m).count("1") % 2 == 0 for m in hf)
    assert 0 in hf
    assert not any(m & opt.bit(7) for m in hf) if n_gpus == 1 else True


@pytest.mark.parametrize("n_gpus", [1, 4])
def test_main_effects_and_two_way_not_aliased(n_gpus):
    act = opt.active_numbers(n_gpus)
    X, names = model_matrix(half_fraction(act), act, interactions=True)
    # Orthogonal columns => X'X diagonal: no main effect or 2-way interaction aliased with another.
    xtx = X.T @ X
    assert np.allclose(xtx, np.diag(np.diag(xtx)))
    assert np.linalg.matrix_rank(X) == len(names) == 1 + len(act) + len(list(combinations(act, 2)))


def test_full_plan_main(exp_full, env_main):
    pts = all_points(exp_full.regimes)
    cells = build_cells(pts, opt.active_numbers(env_main.num_gpus), "full")
    assert len(cells) == 512
    per = {}
    for c in cells:
        per[c.point_id] = per.get(c.point_id, 0) + 1
    assert per["center"] == 128
    assert all(v == 64 for k, v in per.items() if k != "center") and len(per) == 7
    for p in per:
        assert any(c.mask == 0 and c.point_id == p for c in cells)


def test_points(exp_full):
    pts = all_points(exp_full.regimes)
    assert [p.point_id for p in pts][0] == "center"
    assert len(pts) == 7
    for p in pts[1:]:
        # exactly one regime off-centre
        diffs = [(p.context != pts[0].context), (p.concurrency != pts[0].concurrency), (p.reuse != pts[0].reuse)]
        assert sum(diffs) == 1


def test_smoke_plan():
    s = load_settings("smoke", "test")
    act = opt.active_numbers(s.env.num_gpus)
    cfgs = smoke_configs(act)
    assert len(cfgs) == 8 and cfgs[0] == 0
    on = set()
    for m in cfgs:
        on |= set(opt.on_numbers(m))
    assert on == {1, 2, 3, 4, 5, 6}
