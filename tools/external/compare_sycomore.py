"""Optional external check: the RF-spoiled steady state of the engine against sycomore, another group's simulator.

sycomore (Lamy and others, MIT licence, https://github.com/lamyj/sycomore) implements extended phase graphs in C++
with time stepping. It is not a dependency of mrgym; install it for this check only:

    pip install sycomore
    python tools/external/compare_sycomore.py            # prints the table, exits 1 above the bounds
    python tools/external/compare_sycomore.py --write    # also stores docs/physics/external_sycomore.json

Thirteen cases: flip 10 to 60 degrees, TR 5 to 20 ms, RF phase increments 0 / 50 / 117 / 150 degrees, seven without
diffusion and six with diffusion (0.7 to 2.2 um^2/ms, spoiler 20 to 600 mT*ms/m). What it shows: two implementations
written independently agree on the EPG steady state. With diffusion both use the constant-gradient b-factors of
Weigel 2010, so the comparison checks the implementation, not the diffusion model.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

from mrgym.scanner import epg, kspace

CASES = [  # flip_deg, tr_ms, t1_ms, t2_ms, phi_deg, adc_um2_ms, spoiler_mt_ms_per_m
    (10, 5.0, 1000, 80, 117.0, 0.0, 100.0), (30, 5.0, 1000, 80, 117.0, 0.0, 100.0), (30, 5.0, 1500, 200, 117.0, 0.0, 100.0),
    (45, 10.0, 600, 40, 50.0, 0.0, 100.0), (20, 15.0, 1000, 80, 150.0, 0.0, 100.0), (60, 20.0, 2500, 150, 117.0, 0.0, 100.0),
    (30, 5.0, 1000, 80, 0.0, 0.0, 100.0),
    (30, 5.0, 1000, 80, 117.0, 1.0, 100.0), (30, 5.0, 1500, 200, 117.0, 2.0, 300.0), (20, 15.0, 1000, 80, 117.0, 1.0, 20.0),
    (60, 20.0, 2500, 150, 117.0, 2.2, 600.0), (45, 10.0, 600, 40, 50.0, 0.7, 250.0), (30, 5.0, 1000, 80, 0.0, 1.0, 300.0),
]
BOUND_NO_DIFFUSION, BOUND_DIFFUSION = 1e-10, 1e-7


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    try:
        import sycomore
        from sycomore.units import deg, m, ms, mT, um
    except ImportError:
        print("sycomore is not installed (pip install sycomore); nothing was compared")
        return 2
    rows = []
    for flip, tr, t1, t2, phi, adc, area in CASES:
        bd = kspace.diffusion_bd(adc, area, tr) if adc else 0.0
        ours = abs(epg.steady_state(np.deg2rad(flip), tr, t1, t2, phi, (1,), bd)[0])
        model = sycomore.epg.Regular(sycomore.Species(t1 * ms, t2 * ms, adc * um ** 2 / ms), unit_gradient_area=area * mT / m * ms)
        model.threshold = 1e-12
        gradient = (area * mT / m * ms) / (tr * ms)
        echo = 0j
        for r in range(int(np.ceil(14 * t1 / tr)) + 20):
            model.apply_pulse(flip * deg, ((phi * r * (r + 1) / 2) % 360.0) * deg)
            echo = model.echo
            model.apply_time_interval(tr * ms, gradient)
        rows.append(dict(flip_deg=flip, tr_ms=tr, t1_ms=t1, t2_ms=t2, rf_phase_increment_deg=phi, adc_um2_ms=adc,
                         spoiler_mt_ms_per_m=area, mrgym=float(ours), sycomore=float(abs(echo)), abs_diff=float(abs(ours - abs(echo)))))
    nd = max(r["abs_diff"] for r in rows if not r["adc_um2_ms"])
    wd = max(r["abs_diff"] for r in rows if r["adc_um2_ms"])
    print(f"{'flip':>5}{'TR':>6}{'T1':>6}{'T2':>5}{'phi':>6}{'ADC':>5}{'spoiler':>8}{'mrgym':>12}{'sycomore':>12}{'|diff|':>10}")
    for r in rows:
        print(f"{r['flip_deg']:>5}{r['tr_ms']:>6}{r['t1_ms']:>6}{r['t2_ms']:>5}{r['rf_phase_increment_deg']:>6}{r['adc_um2_ms']:>5}"
              f"{r['spoiler_mt_ms_per_m']:>8}{r['mrgym']:>12.8f}{r['sycomore']:>12.8f}{r['abs_diff']:>10.1e}")
    ok = nd <= BOUND_NO_DIFFUSION and wd <= BOUND_DIFFUSION
    print(f"max |difference| in units of M0: {nd:.1e} without diffusion (bound {BOUND_NO_DIFFUSION:g}), "
          f"{wd:.1e} with diffusion (bound {BOUND_DIFFUSION:g}): {'ok' if ok else 'FAIL'}")
    if a.write:
        import mrgym
        out = dict(sycomore_version=getattr(sycomore, "__version__", "unknown"), engine=mrgym.ENGINE_ID,
                   date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), cases=len(rows),
                   max_abs_diff_no_diffusion=nd, max_abs_diff_with_diffusion=wd, bounds=[BOUND_NO_DIFFUSION, BOUND_DIFFUSION],
                   passed=bool(ok), command="pip install sycomore && python tools/external/compare_sycomore.py --write")
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "physics", "external_sycomore.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print("written:", os.path.relpath(path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
