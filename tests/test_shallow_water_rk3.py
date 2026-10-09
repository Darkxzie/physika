"""Numerical checks of the compiled periodic, wet-bed shallow-water tutorial.

No solver is reimplemented here. The Fourier shear test has an exact
semidiscrete solution, so its time-order check contains no spatial error.
"""

from pathlib import Path

import pytest
import torch

import physika.runtime
from physika.codegen import from_ast_to_torch
from physika.core.elab.elab import Elab
from physika.core.inductive import mk_builtin_env
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.units import dim_analysis
from physika.utils.ast_utils import build_unified_ast
from physika.utils.import_manager import resolve_imports


SOURCE = Path(__file__).resolve().parents[1] / "tutorials/shallow_water_rk3.phyk"


@pytest.fixture(scope="module")
def model():
    """Use the CLI's elaboration/codegen path, with isolated runtime globals."""
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        symbol_table.clear()
        lexer.lexer.lineno = 1
        parsed = parser.parse(SOURCE.read_text(encoding="utf-8"), lexer=lexer)
        parsed = resolve_imports(parsed, SOURCE)
        unified = build_unified_ast(parsed, symbol_table)
        elaborator = Elab(mk_builtin_env())
        resolved = elaborator.elaborate(unified)
        signatures = {}
        assert not dim_analysis(unified, elaborator.state.env, signatures)
        code = from_ast_to_torch(
            unified,
            print_code=False,
            resolved_bodies=resolved.get("resolved_bodies"),
            resolved_methods=resolved.get("resolved_methods"),
            resolved_program=resolved.get("resolved_program"),
            resolved_program_fvar_names=resolved.get("resolved_program_fvar_names"),
            cic_env=elaborator.state.env,
            func_sigs=signatures,
        )
        namespace = vars(physika.runtime).copy()
        exec(compile(code, str(SOURCE) + " (generated)", "exec"), namespace)
        yield namespace
    finally:
        torch.set_default_dtype(previous_dtype)
        symbol_table.clear()
        lexer.lexer.lineno = 1


def advance(model, q, g, dx, dy, dt, steps):
    for _ in range(steps):
        q = model["rk3_step"](q, g, dx, dy, dt)
    return q


def smooth_state(nx=13, ny=9):
    x = 2 * torch.pi * (torch.arange(nx) + 0.5) / nx
    y = 2 * torch.pi * (torch.arange(ny) + 0.5) / ny
    h = 1 + 0.08 * torch.cos(x[:, None]) * torch.sin(y[None, :])
    u = 0.2 + 0.04 * torch.sin(x[:, None] + y[None, :])
    v = -0.1 + 0.03 * torch.cos(2 * x[:, None] - y[None, :])
    return torch.stack((h, h * u, h * v))


@pytest.mark.parametrize("u,v", [(0.0, 0.0), (0.3, -0.2)])
def test_constant_resting_and_moving_states(model, u, v):
    h = torch.full((11, 7), 1.4)
    q = torch.stack((h, h * u, h * v))
    torch.testing.assert_close(model["rhs"](q, 2.1, 0.13, 0.21),
                               torch.zeros_like(q), rtol=0, atol=1e-14)
    result = advance(model, q, 2.1, 0.13, 0.21, 0.002, 30)
    torch.testing.assert_close(result, q, rtol=0, atol=2e-14)


@pytest.mark.parametrize("axis", [0, 1])
def test_face_speed_uses_both_neighbors_including_periodic_seam(model, axis):
    speeds = torch.tensor([[0.1, 2.0, 0.4], [3.0, 0.2, 1.0]])
    expected = torch.maximum(speeds, torch.roll(speeds, -1, axis))
    torch.testing.assert_close(model["face_speed"](speeds, axis), expected)


def test_periodic_mass_and_both_momenta_are_conserved(model):
    q = smooth_state()
    rhs = model["rhs"](q, 1.7, 1 / 13, 1 / 9)
    torch.testing.assert_close(rhs.sum(dim=(1, 2)), torch.zeros(3),
                               rtol=0, atol=2e-13)
    result = advance(model, q, 1.7, 1 / 13, 1 / 9, 0.001, 80)
    torch.testing.assert_close(result.sum(dim=(1, 2)), q.sum(dim=(1, 2)),
                               rtol=0, atol=2e-12)
    assert torch.isfinite(result).all()
    assert result[0].min() > 0
    assert (result[1:] - q[1:]).abs().amax(dim=(1, 2)).min() > 1e-3


def test_axis_exchange_on_rectangular_grid(model):
    q = smooth_state()
    exchanged = q[[0, 2, 1]].transpose(1, 2)
    direct = advance(model, q, 1.7, 0.08, 0.13, 0.002, 8)
    swapped = advance(model, exchanged, 1.7, 0.13, 0.08, 0.002, 8)
    torch.testing.assert_close(swapped, direct[[0, 2, 1]].transpose(1, 2),
                               rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tutorial_all_stages_are_wet_finite_and_within_cfl(model, monkeypatch, dtype):
    """Audit the real RHS inputs, not separately reconstructed RK stages."""
    original_rhs = model["rhs"]
    dt, g, dx, dy = (model[key] for key in ("Δt", "g", "dx", "dy"))
    stage_count = 0
    minimum_depth = float("inf")
    maximum_cfl = 0.0

    def check(q):
        nonlocal minimum_depth, maximum_cfl
        assert q.dtype == dtype
        assert torch.isfinite(q).all()
        h = q[0]
        minimum_depth = min(minimum_depth, float(h.min()))
        assert h.min() > 0
        c = torch.sqrt(g * h)
        cfl = dt * ((q[1] / h).abs().add(c).max() / dx
                    + (q[2] / h).abs().add(c).max() / dy)
        maximum_cfl = max(maximum_cfl, float(cfl))
        assert cfl <= 0.2

    def audited_rhs(q, gravity, spacing_x, spacing_y):
        nonlocal stage_count
        check(q)
        stage_count += 1
        return original_rhs(q, gravity, spacing_x, spacing_y)

    monkeypatch.setitem(model, "rhs", audited_rhs)
    q = model["initial"].to(dtype=dtype).clone()
    initial_mass = q[0].sum()
    for _ in range(int(model["steps"])):
        q = model["rk3_step"](q, g, dx, dy, dt)
        check(q)
    assert stage_count == 3 * int(model["steps"])
    if dtype == torch.float64:
        torch.testing.assert_close(q, model["final"], rtol=0, atol=0)
        torch.testing.assert_close(q[0].sum(), initial_mass, rtol=0, atol=5e-12)
    else:
        torch.testing.assert_close(q.double(), model["final"], rtol=0, atol=3e-6)
        torch.testing.assert_close(q[0].sum(), initial_mass, rtol=1e-6, atol=0)
    assert q[1].abs().max() > 1e-3
    assert q[2].abs().max() > 1e-3
    tolerance = 2e-14 if dtype == torch.float64 else 5e-7
    torch.testing.assert_close(q, q[[0, 2, 1]].transpose(1, 2),
                               rtol=0, atol=tolerance)
    print(f"tutorial {dtype}: min(h)={minimum_depth:.12g}, max CFL={maximum_cfl:.12g}, "
          f"mass drift={float(q[0].sum() - initial_mass):.3e}, stages={stage_count}")


@pytest.mark.parametrize("axis", [0, 1])
def test_pressure_accelerates_downhill_in_both_directions(model, axis):
    """At rest momentum changes by the centered hydrostatic pressure gradient."""
    n = 32
    theta = 2 * torch.pi * (torch.arange(n) + 0.5) / n
    profile = 1.2 + 0.05 * torch.cos(theta)
    h = profile[:, None].expand(n, 5)
    if axis == 1:
        h = h.T
    q = torch.stack((h, torch.zeros_like(h), torch.zeros_like(h)))
    g, spacing = 2.3, 1 / n
    rate = model["rhs"](q, g, spacing, spacing)
    pressure = 0.5 * g * h.square()
    expected = -(torch.roll(pressure, -1, axis)
                 - torch.roll(pressure, 1, axis)) / (2 * spacing)
    torch.testing.assert_close(rate[1 + axis], expected, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(rate[2 - axis], torch.zeros_like(h), rtol=0, atol=0)
    assert rate[1 + axis].abs().max() > 0.5


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("speed", [-0.4, 0.4])
def test_advected_shear_flux_and_third_order_time_integration(model, axis, speed):
    """Exact Fourier eigenmode: constant depth and normal speed, transverse shear.

    lambda = -i U sin(theta)/dx + (|U|+sqrt(g H))(cos(theta)-1)/dx.
    This physical contact wave isolates transport and Rusanov damping without
    duplicating the flux routine or using a numerical reference integrator.
    """
    n, h, g, amplitude = 24, 1.3, 1.7, 0.07
    dx = 1 / n
    angle = torch.tensor(2 * torch.pi / n)
    phase = angle * (torch.arange(n) + 0.5)
    depth = torch.full((n, 5), h)
    shear = amplitude * torch.cos(phase)[:, None].expand(n, 5)
    q = torch.stack((depth, depth * speed, shear))
    if axis == 1:
        q = q[[0, 2, 1]].transpose(1, 2)
    decay = (abs(speed) + (g * h) ** 0.5) * (torch.cos(angle) - 1) / dx
    frequency = speed * torch.sin(angle) / dx

    def shear_at(time):
        values = (amplitude * torch.exp(decay * time)
                  * torch.cos(phase - frequency * time))[:, None].expand(n, 5)
        return values if axis == 0 else values.T

    expected_rate = amplitude * (decay * torch.cos(phase)
                                 + frequency * torch.sin(phase))
    expected_rate = expected_rate[:, None].expand(n, 5)
    if axis == 1:
        expected_rate = expected_rate.T
    rhs = model["rhs"](q, g, dx, dx)
    torch.testing.assert_close(rhs[2 - axis], expected_rate, rtol=2e-12, atol=2e-14)
    torch.testing.assert_close(rhs[0], torch.zeros_like(q[0]), rtol=0, atol=0)
    torch.testing.assert_close(rhs[1 + axis], torch.zeros_like(q[0]), rtol=0, atol=0)

    end_time = 0.08
    errors = []
    for steps in (16, 32, 64):
        result = advance(model, q, g, dx, dx, end_time / steps, steps)
        errors.append(float((result[2 - axis] - shear_at(end_time)).square().mean().sqrt()))
    orders = [float(torch.log2(torch.tensor(a / b)))
              for a, b in zip(errors, errors[1:])]
    print(f"shear axis {axis}, speed {speed}: errors={errors}, temporal orders={orders}")
    assert all(2.9 < order < 3.1 for order in orders)


def test_nonlinear_coupled_temporal_self_convergence(model):
    """Fixed-grid refinement exercises all nonlinear RK stage dependencies."""
    q = smooth_state(12, 10)
    results = [advance(model, q, 1.3, 1 / 12, 1 / 10, 0.08 / steps, steps)
               for steps in (16, 32, 64)]
    differences = [float((a - b).square().mean().sqrt())
                   for a, b in zip(results, results[1:])]
    order = float(torch.log2(torch.tensor(differences[0] / differences[1])))
    print(f"nonlinear refinement: differences={differences}, temporal order={order}")
    assert 2.8 < order < 3.2
