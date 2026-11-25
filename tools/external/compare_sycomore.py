"""External check: the RF-spoiled steady state of the engine against sycomore, another group's simulator.

sycomore (Lamy and others, MIT licence, https://github.com/lamyj/sycomore) implements extended phase graphs in C++
with time stepping. It is not a dependency of mrgym; install it for this check only (binary wheels exist for
Python 3.10):

    pip install sycomore
    python tools/external/compare_sycomore.py            # prints the table, exits 1 above the bounds
    python tools/external/compare_sycomore.py --write    # also stores the numbers under docs/physics/

Seventeen cases. Spoiled gradient echo, thirteen: flip 10 to 60 degrees, TR 5 to 20 ms, RF phase increments 0 / 50 /
117 / 150 degrees, seven without diffusion and six with diffusion (0.7 to 2.2 um^2/ms, spoiler 20 to 600 mT*ms/m).
Dual-TR (actual flip-angle imaging), four: TR1 20 or 30 ms, TR2 = 5 TR1, with and without diffusion. What it shows:
two implementations written independently agree on the EPG steady state. With diffusion both use the
constant-gradient b-factors of Weigel 2010, so the comparison checks the implementation, not the diffusion model.

``--write`` stores the summary (``docs/physics/external_sycomore.json``) and sycomore's value for every case
(``docs/physics/external/sycomore_reference.json``). ``tools/external/run_all.py`` compares the engine with the
stored values, so the comparison can be repeated on a machine without sycomore.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

from mrgym.scanner import epg, kspace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPGR_CASES = [  # flip_deg, tr_ms, t1_ms, t2_ms, phi_deg, adc_um2_ms, spoiler_mt_ms_per_m
    (10, 5.0, 1000, 80, 117.0, 0.0, 100.0), (30, 5.0, 1000, 80, 117.0, 0.0, 100.0), (30, 5.0, 1500, 200, 117.0, 0.0, 100.0),
    (45, 10.0, 600, 40, 50.0, 0.0, 100.0), (20, 15.0, 1000, 80, 150.0, 0.0, 100.0), (60, 20.0, 2500, 150, 117.0, 0.0, 100.0),
    (30, 5.0, 1000, 80, 0.0, 0.0, 100.0),
    (30, 5.0, 1000, 80, 117.0, 1.0, 100.0), (30, 5.0, 1500, 200, 117.0, 2.0, 300.0), (20, 15.0, 1000, 80, 117.0, 1.0, 20.0),
    (60, 20.0, 2500, 150, 117.0, 2.2, 600.0), (45, 10.0, 600, 40, 50.0, 0.7, 250.0), (30, 5.0, 1000, 80, 0.0, 1.0, 300.0),
]
AFI_CASES = [  # flip_deg, tr1_ms, tr_ratio, t1_ms, t2_ms, phi_deg, adc_um2_ms, spoiler_mt_ms_per_m (per TR1)
    (60, 20.0, 5, 1000, 80, 129.3, 0.0, 300.0), (45, 30.0, 5, 1500, 150, 45.0, 0.0, 585.0),
    (60, 20.0, 5, 1000, 80, 129.3, 1.0, 300.0), (70, 30.0, 5, 838, 75, 45.0, 0.75, 585.0),
]
BOUND_NO_DIFFUSION, BOUND_DIFFUSION = 1e-10, 1e-7


def engine_spgr(flip, tr, t1, t2, phi, adc, area):
    bd = kspace.diffusion_bd(adc, area, tr) if adc else 0.0
    return [float(abs(epg.steady_state(np.deg2rad(flip), tr, t1, t2, phi, (1,), bd)[0]))]


def engine_afi(flip, tr1, n, t1, t2, phi, adc, area):
    bd = kspace.diffusion_bd(adc, area, tr1) if adc else 0.0
    return [float(abs(v)) for v in epg.steady_state(np.deg2rad(flip), tr1, t1, t2, phi, (1, 1) + (0,) * (n - 1), bd)]


def sycomore_values():
    """sycomore's echo magnitudes for every case: (spgr [[s]], afi [[s1, s2]]). Raises ImportError without sycomore."""
    import sycomore
    from sycomore.units import deg, m, ms, mT, um

    def run(flip, tr, t1, t2, phi, adc, area, pattern):
        model = sycomore.epg.Regular(sycomore.Species(t1 * ms, t2 * ms, adc * um ** 2 / ms), unit_gradient_area=area * mT / m * ms)
        model.threshold = 1e-12
        gradient, p, echo = (area * mT / m * ms) / (tr * ms), len(pattern), {}
        for j in range((int(np.ceil(14 * t1 / (tr * p))) + 20) * p):
            if pattern[j % p]:
                model.apply_pulse(flip * deg, ((phi * j * (j + 1) / 2) % 360.0) * deg)
                echo[j % p] = abs(model.echo)
            model.apply_time_interval(tr * ms, gradient)
        return [float(echo[k]) for k in sorted(echo)]
    return ([run(*c, pattern=(1,)) for c in SPGR_CASES],
            [run(c[0], c[1], *c[3:], pattern=(1, 1) + (0,) * (c[2] - 1)) for c in AFI_CASES])


def compare(spgr_ref, afi_ref):
    """Rows of |engine - reference| and the two largest differences (without, with diffusion)."""
    rows = []
    for c, ref in zip(SPGR_CASES, spgr_ref):
        ours = engine_spgr(*c)
        rows.append(dict(sequence="spgr", case=list(c), mrgym=ours, sycomore=ref, abs_diff=float(np.max(np.abs(np.array(ours) - np.array(ref)))), diffusion=bool(c[5])))
    for c, ref in zip(AFI_CASES, afi_ref):
        ours = engine_afi(*c)
        rows.append(dict(sequence="afi", case=list(c), mrgym=ours, sycomore=ref, abs_diff=float(np.max(np.abs(np.array(ours) - np.array(ref)))), diffusion=bool(c[6])))
    return rows, max(r["abs_diff"] for r in rows if not r["diffusion"]), max(r["abs_diff"] for r in rows if r["diffusion"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    try:
        import sycomore
        spgr_ref, afi_ref = sycomore_values()
    except ImportError:
        print("sycomore is not installed (pip install sycomore); nothing was compared")
        return 2
    rows, nd, wd = compare(spgr_ref, afi_ref)
    for r in rows:
        print(f"{r['sequence']:5s} {str(r['case']):52s} mrgym {r['mrgym'][0]:.10f}  sycomore {r['sycomore'][0]:.10f}  |diff| {r['abs_diff']:.1e}")
    ok = nd <= BOUND_NO_DIFFUSION and wd <= BOUND_DIFFUSION
    print(f"max |difference| in units of M0: {nd:.1e} without diffusion (bound {BOUND_NO_DIFFUSION:g}), "
          f"{wd:.1e} with diffusion (bound {BOUND_DIFFUSION:g}): {'ok' if ok else 'FAIL'}")
    if a.write:
        import mrgym
        version = getattr(sycomore, "__version__", None)
        if version is None:
            import importlib.metadata
            version = importlib.metadata.version("sycomore")
        date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        out = dict(sycomore_version=version, engine=mrgym.ENGINE_ID, date_utc=date, cases=len(rows),
                   max_abs_diff_no_diffusion=nd, max_abs_diff_with_diffusion=wd, bounds=[BOUND_NO_DIFFUSION, BOUND_DIFFUSION],
                   passed=bool(ok), command="pip install sycomore && python tools/external/compare_sycomore.py --write")
        ref = dict(sycomore_version=version, python=sys.version.split()[0], date_utc=date, quantity="|echo| right after the pulse, units of M0",
                   spgr_cases=[list(c) for c in SPGR_CASES], spgr=spgr_ref, afi_cases=[list(c) for c in AFI_CASES], afi=afi_ref)
        for rel, obj in (("docs/physics/external_sycomore.json", out), ("docs/physics/external/sycomore_reference.json", ref)):
            path = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=1)
                f.write("\n")
            print("written:", rel)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
