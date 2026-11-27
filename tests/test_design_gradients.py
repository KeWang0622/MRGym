"""Gradient programs: what a program is, rasterising, the limit report, the lobe designers' argument checks."""
import numpy as np
import pytest

from mrgym.design import gradients as G
from mrgym.design import pulses


def test_program_rules():
    assert G.program(x=[(0, 0), (1, 5)]) == {"x": ((0.0, 0.0), (1.0, 5.0))}
    for bad in (dict(w=[(0, 0)]), dict(x=[(0, 0), (0, 1)]), dict(x=[(1, 0), (0.5, 1)]), dict(x=[(-1, 0)]), dict(x=[(0, float("nan"))]),
                dict(x=[(i * 0.1, 1.0) for i in range(33)]), dict(x=[]), {}):
        with pytest.raises(ValueError):
            G.program(**bad)
    assert len(G.program(x=[(i * 0.1, 1.0) for i in range(32)])["x"]) == 32


def test_rasterise_gives_what_the_hardware_would_play():
    t, g = G.rasterise(G.program(x=[(0.1, 0.0), (0.2, 10.0), (0.4, 10.0), (0.5, 0.0)], z=[(0.0, 0.0), (0.3, 3.0)]))
    assert t.shape == (51,) and g.shape == (51, 3) and abs(t[1] - 0.01) < 1e-15
    assert abs(g[15, 0] - 5.0) < 1e-12 and g[5, 0] == 0.0 and np.all(g[:, 1] == 0.0) and abs(g[30, 2] - 3.0) < 1e-12 and g[31, 2] == 0.0
    t, g = G.rasterise(G.program(x=[(0.0, 0.0), (0.005, 10.0), (0.01, 0.0)]))            # a spike between two raster times
    assert np.all(g == 0.0)
    assert G.rasterise(G.program(x=[(0, 0), (0.1, 1)]), t_end_ms=1.0)[0][-1] == 1.0
    assert G.on_raster(G.program(x=[(0.0, 0.0), (0.3, 1.0)])) and not G.on_raster(G.program(x=[(0.0, 0.0), (0.305, 1.0)]))


def test_hardware_record_reads_the_console():
    from mrgym.scanner import console
    hw = G.Hardware.console()
    assert (hw.g_max_mt_m, hw.slew_t_m_s, hw.raster_ms, hw.b1_peak_ut) == tuple(float(console.SYSTEM[k]) for k in
                                                                              ("g_max_mt_m", "slew_t_m_s", "raster_ms", "b1_peak_ut"))
    assert G.Hardware.console(rule="vector").rule == "vector"
    with pytest.raises(ValueError):
        G.Hardware(rule="sphere")


def test_check_reports_every_field_and_the_stimulation_limit():
    t, g = G.rasterise(G.program(x=[(0, 0), (0.3, 40.0), (0.9, -40.0), (1.2, 0)]))
    rep = G.check(t, g)
    assert {"ok", "violations", "rule", "duration_ms", "peak_mt_m", "peak_vector_mt_m", "slew_t_m_s", "slew_vector_t_m_s", "pns_percent",
            "pns_combined_percent", "heating_mt2_m2_ms", "on_raster"} == set(rep)
    assert [v["code"] for v in rep["violations"]] == ["STIMULATION_LIMIT"] and rep["pns_combined_percent"] > 80.0
    assert abs(rep["slew_t_m_s"]["x"] - 40.0 / 0.3) < 1e-6 and rep["on_raster"]
    assert G.check(t, g, G.Hardware(pns_limit_percent=200.0))["ok"]
    assert G.check(t, 0.5 * g)["ok"]                                    # half the slew, the same ramp times
    coarse = G.check(np.arange(5) * 0.02, np.array([0, 1, 2, 1, 0.0]))
    assert [v["code"] for v in coarse["violations"]] == ["NOT_ON_RASTER"]
    with pytest.raises(ValueError):
        G.check(np.array([0.0, 0.0, 0.01]), np.zeros(3))


def test_stimulation_combines_axes_as_a_root_sum_of_squares():
    v = [(0, 0), (0.2, 20.0), (1.0, 20.0), (1.2, 0)]
    one = G.check(*G.rasterise(G.program(x=v)))
    two = G.check(*G.rasterise(G.program(x=v, y=v)))
    assert abs(two["pns_combined_percent"] / one["pns_combined_percent"] - np.sqrt(2)) < 1e-4
    assert two["pns_percent"]["x"] == one["pns_percent"]["x"]


def test_refocusing_changes_the_sign_from_that_time_on():
    t, g = G.rasterise(G.program(x=[(0, 0), (0.1, 10), (1.0, 10), (1.1, 0), (2.0, 0), (2.1, 10), (3.0, 10), (3.1, 0)]))
    area = 10 * 1.0
    assert abs(G.moments(t, g, 0.0, 0)[0, 0] - 2 * area) < 1e-12
    assert abs(G.moments(t, g, 0.0, 0, refocus_ms=(1.5,))[0, 0]) < 1e-12
    assert abs(G.moments(t, g, 0.0, 0, refocus_ms=(1.2, 1.7))[0, 0] - 2 * area) < 1e-12        # two refocusing pulses: the sign is back
    tt, k = G.k_trajectory(t, g, refocus_ms=(1.555,))
    assert 1.555 in tt and abs(k[-1, 0]) < 1e-15 and k[:, 0].max() > 0
    assert G.b_value(t, g, (1.5,)) > 0 and abs(G.b_value(t, g, (1.5,)) - G.b_value(t, -g, (1.5,))) < 1e-12


def test_trapezoid_arguments_and_sign():
    neg = G.trapezoid(-5.0, 40.0, 150.0)
    assert neg["peak_mt_m"] < 0 and neg["duration_ms"] == G.trapezoid(5.0, 40.0, 150.0)["duration_ms"]
    assert G.trapezoid(0.0, 40.0, 150.0)["duration_ms"] == 0.0
    assert G.trapezoid(5.0, 40.0, 150.0, start_ms=2.0)["vertices"][0] == (2.0, 0.0)
    tri = G.trapezoid(1.0, 40.0, 150.0, raster_ms=0.01)
    assert len(tri["vertices"]) == 3 and tri["flat_ms"] == 0.0 and G.on_raster(G.program(x=tri["vertices"]))
    with pytest.raises(ValueError):
        G.trapezoid(5.0, 0.0, 150.0)


def test_optimal_lobe_arguments():
    for bad in (dict(targets={}), dict(targets={0: 0.0}), dict(targets={5: 1.0}), dict(targets={0: 1.0}, ref="middle")):
        args = dict(targets={0: 1.0}, g_max_mt_m=40.0, slew_t_m_s=150.0)
        args.update(bad)
        with pytest.raises(ValueError):
            G.optimal_lobe(**args)
    with pytest.raises(ValueError, match="raster steps"):
        G.optimal_lobe({0: 500.0}, 40.0, 150.0, max_steps=64)
    res = G.optimal_lobe({0: 2.0, 1: 0.0}, 40.0, 150.0, ref="start", offset_ms=0.5)
    assert res["residual"] < 1e-9 and res["g_mt_m"][0] == 0.0 and res["g_mt_m"][-1] == 0.0 and "infeasible" in res["certificate"]
    assert abs(G.moments(res["t_ms"], res["g_mt_m"], -0.5, 1)[1, 0]) < 1e-9


def test_pulse_gradient_check():
    p = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    assert G.check_pulse_gradient(p)["ok"]
    thin = pulses.sinc(90.0, 1.0, 8.0, 1.0)                              # 188 mT/m, 46 uT
    assert [v["code"] for v in G.check_pulse_gradient(thin)["violations"]] == ["GRADIENT_LIMIT", "B1_PEAK_LIMIT"]
    fast = pulses.verse(p, 6.0, 20.0)
    assert [v["code"] for v in G.check_pulse_gradient(fast, G.Hardware(slew_t_m_s=150.0))["violations"]] == ["SLEW_LIMIT"]


def test_a_last_vertex_on_a_raster_time_is_played_whatever_its_last_bit():
    """k x 0.01 is not always the double nearest to the decimal time (57 x 0.01 > 0.57): a program that ends above
    zero must be reported as such for every end time."""
    for k in range(1, 400):
        end = k / 100
        t, g = G.rasterise(G.program(x=[(0.0, 0.0), (end, 1.0)]))
        assert abs(g[-1, 0] - 1.0) < 1e-12 and t.size == k + 1, k
        assert [v["code"] for v in G.check(t, g, G.Hardware(slew_t_m_s=1e9, pns_limit_percent=1e9))["violations"]] == ["NOT_ZERO_AT_ENDS"], k
    t, g = G.rasterise(G.program(x=[(0.57, 2.0), (0.9, 0.0)]))            # the same at the first vertex
    assert abs(g[57, 0] - 2.0) < 1e-12 and g[56, 0] == 0.0


def test_the_axis_named_carries_the_value_reported():
    t = np.arange(4) * 0.01
    g = np.zeros((4, 3))
    g[1] = (30.0, 35.0, 0.0)
    v = G.check(t, g, G.Hardware(g_max_mt_m=29.0, slew_t_m_s=1e9, pns_limit_percent=1e9))["violations"][0]
    assert (v["code"], v["axis"], v["value"]) == ("GRADIENT_LIMIT", "y", 35.0)
    g[1] = (35.0, 30.0, 0.0)
    v = G.check(t, g, G.Hardware(g_max_mt_m=29.0, slew_t_m_s=1e9, pns_limit_percent=1e9))["violations"][0]
    assert (v["axis"], v["value"]) == ("x", 35.0)


def test_lobe_designers_at_their_edges():
    assert G.optimal_lobe({0: 5.0}, 40.0, 150.0, max_steps=40)["steps"] == 37             # a cap that is not a power of two
    with pytest.raises(ValueError, match="raster steps"):
        G.optimal_lobe({0: 5.0}, 40.0, 150.0, max_steps=36)
    tiny = G.trapezoid(1e-22, 40.0, 150.0, raster_ms=0.01)
    assert tiny["duration_ms"] == 0.02 and abs(tiny["peak_mt_m"] - 1e-20) < 1e-30
    with pytest.raises(ValueError, match="orders 0 to 4"):
        G.moments(np.arange(3) * 0.01, np.array([0.0, 1.0, 0.0]), 0.0, order=5)
    assert G.rasterise(G.program(x=[(0, 0)]))[0].size == 2
