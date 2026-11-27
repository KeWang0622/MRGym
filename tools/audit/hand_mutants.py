"""Hand mutants: one deliberate defect at a time, applied to a copy of ``src/``, and the test groups that notice.

M01 to M16 are the sixteen edits of the engineering review of 2026-11-24 (its mutation spot checks): eight wrong-physics edits,
six scoring / pricing / budget edits, and two edits that are valid physics but other data (noise order, phantom
fields). M17 to M19 are edits of module-level constants of the family code, where a mutation tool does not reach. M08, M09, M11 and
M12 were written against the two v0 environments; they are the same defects in the code of their v1 successors. M20 to M30
are further edits of the two v1 families (pass rules, constants, the oracle).
M31 to M33 are defects of the slice-selective path of engine 0.2.0. M34 to M38 undo, one at a time, the corrections of
engine 0.3.0 (homodyne weights, fat saturation line by line, the recovery after the saturation pulse, the calibration
of the slice gradient) and flip the sign of the centre-frequency adjustment. M39 and M40 put back the sign of the
``bssfp`` phase cycle of engine 0.2.0, for hard pulses and for a slice.

    python tools/audit/hand_mutants.py                 all mutants, 2 at a time; exit 1 unless every one is killed
    python tools/audit/hand_mutants.py --only M07 M11  some of them
    python tools/audit/hand_mutants.py --json out.json the table as JSON (for the audit report)

Each mutant is run against three groups of tests separately, so the table shows which line of defence notices:
  scoring       the scoring tables (rows and episodes), the gate, the family contract, the task snapshots
  metamorphic   tests/test_metamorphic.py
  golden        tests/test_golden.py
A mutant is killed when at least one group fails. The unmodified tree is run first and must pass every group.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = dict(
    scoring=["tests/test_scoring_table.py", "tests/test_scoring_episodes.py", "tests/test_null_tissue.py", "tests/test_nowrap.py",
             "tests/test_t1_transmit.py", "tests/test_gate.py", "tests/test_family_contract.py", "tests/test_task_snapshots.py"],
    metamorphic=["tests/test_metamorphic.py"],
    golden=["tests/test_golden.py"],
)
S = "src/mrgym/scanner/"
E = "src/mrgym/envs/"
# (id, kind, what, file, old, new); ``old`` must occur exactly once in the file
MUTANTS = [
    ("M01", "physics", "EPG recovery term uses E2 instead of E1", S + "epg.py",
     "    b[4] = 1 - e1 ", "    b[4] = 1 - e2 "),
    ("M02", "physics", "spin echo: sin(beta / 2) instead of sin^2(beta / 2)", S + "signal.py",
     "events = se_events(a, tr_ms, te_ms)\n"
     "    mz = mz_steady(events, t1_ms) if sat is None else mz_saturated(events, t1_ms, sat[0], sat[1])\n"
     "    return mz * np.sin(a) * np.sin(a) ** 2",
     "events = se_events(a, tr_ms, te_ms)\n"
     "    mz = mz_steady(events, t1_ms) if sat is None else mz_saturated(events, t1_ms, sat[0], sat[1])\n"
     "    return mz * np.sin(a) * np.sin(a)"),
    ("M03", "physics", "inversion recovery forgets TI in the recovery period", S + "signal.py",
     '("wait", tr_ms - ti_ms - te_ms / 2)', '("wait", tr_ms - te_ms / 2)'),
    ("M04", "physics", "averaging does not reduce the noise", S + "kspace.py",
     'sd = sy["sigma0"] * np.sqrt(k_ro * rp["bandwidth_hz_per_px"] / (nex * rp.get("partitions", 1)))\n                ks[i]',
     'sd = sy["sigma0"] * np.sqrt(k_ro * rp["bandwidth_hz_per_px"] / rp.get("partitions", 1))\n                ks[i]'),
    ("M05", "physics", "off-resonance shift along the readout has the wrong sign", S + "kspace.py",
     'shift, axis = -rp["ro_polarity"] * df / rp["bandwidth_hz_per_px"]', 'shift, axis = rp["ro_polarity"] * df / rp["bandwidth_hz_per_px"]'),
    ("M06", "physics", "gradient echo decays with T2, not T2*", S + "signal.py",
     'f = np.exp(-t_ms / cls["t2s_ms"])', 'f = np.exp(-t_ms / cls["t2_ms"])'),
    ("M07", "scoring", "scan time ignores partial Fourier (lines_full for lines)", S + "console.py",
     'n_echo=1, shots=lines, echo_times=None', 'n_echo=1, shots=lines_full, echo_times=None'),     # the cycles per average of a line-by-line scan
    ("M08", "scoring", 'the "at least one scan" gate dropped from the reward (NoWrap-v0 had its own; every family shares this one)', E + "family.py",
     'measured=bool(rep["scans"] > 0))', 'measured=True)'),
    ("M09", "scoring", "T1Transmit: the error of a region is signed (a T1 far too short passes)", E + "t1_transmit/task.py",
     'miss = {i: abs(math.log(float(answer["t1_ms"][i]) / t)) for i, t in truth.items()}',
     'miss = {i: math.log(float(answer["t1_ms"][i]) / t) for i, t in truth.items()}'),
    ("M10", "scoring", "budget check against the total, not what is left", S + "session.py",
     "if cost_s > self.remaining + 1e-9:", "if cost_s > self.total + 1e-9:"),
    ("M11", "scoring", "NoWrap: wrap-around limit ten times looser", E + "nowrap/task.py",
     "WRAP_LIMIT = 2e-3 ", "WRAP_LIMIT = 2e-2 "),
    ("M12", "scoring", "NoWrap: the object has to be inside along one image axis only (min for max)", E + "nowrap/task.py",
     "        out += [lo - (float(p.min()) - half), (float(p.max()) + half) - hi]\n    return max(out)",
     "        out += [lo - (float(p.min()) - half), (float(p.max()) + half) - hi]\n    return min(out)"),
    ("M13", "physics", "bSSFP numerator uses E1 for E2", S + "signal.py",
     "(1 - e1) * np.sin(a) * (1 - e2 * np.exp(-1j * theta))", "(1 - e1) * np.sin(a) * (1 - e1 * np.exp(-1j * theta))"),
    ("M14", "physics", "transmit-gain error always positive", S + "world.py",
     "dev = float(rng.uniform(lo, hi)) * (1 if rng.uniform() < 0.5 else -1)", "dev = float(rng.uniform(lo, hi)) * (1 if rng.uniform() < 2.0 else -1)"),
    ("M15", "other data", "noise drawn in another order (same statistics, other data)", S + "kspace.py",
     "ks[i] += sd * (rng.standard_normal((k_pe, k_ro)) + 1j * rng.standard_normal((k_pe, k_ro)))",
     "ks[i] += sd * (1j * rng.standard_normal((k_pe, k_ro)) + rng.standard_normal((k_pe, k_ro)))"),
    ("M16", "other data", "receive-profile range of the brain / abdomen phantoms changed", S + "world.py",
     "rxm = 1 + rng.uniform(0.1, 0.3) * rho2", "rxm = 1 + rng.uniform(0.1, 0.4) * rho2"),
    ("M17", "constant", 'family rule: the "measured" gate dropped', E + "family.py",
     'GATES = ("log_replays_within_budget", "measured")', 'GATES = ("log_replays_within_budget",)'),
    ("M18", "constant", "NullTissue level 1: pass threshold ten times looser", E + "null_tissue/task.py",
     'tr_ms=(6000.0, 6000.0), n_regions=2,\n            null_max=0.05, budget_s=150.0', 'tr_ms=(6000.0, 6000.0), n_regions=2,\n            null_max=0.5, budget_s=150.0'),
    ("M19", "constant", "gate: a must-fail policy may pass twice as often", "src/mrgym/gate.py",
     "ALPHA, REF_MIN, BLIND_MAX, SEP_MIN = 0.05, 0.90, 0.10, 0.25", "ALPHA, REF_MIN, BLIND_MAX, SEP_MIN = 0.05, 0.90, 0.20, 0.25"),
    ("M20", "constant", "NoWrap: a scan may take 50 % longer than the fastest (1.10 -> 1.50)", E + "nowrap/task.py",
     "TOL = 1.10 ", "TOL = 1.50 "),
    ("M21", "constant", "NoWrap: the object may stick out of the image by 10 mm", E + "nowrap/task.py",
     "COVER_SLACK_MM = 0.5", "COVER_SLACK_MM = 10.0"),
    ("M22", "scoring", "NoWrap: the scan-time check dropped from the pass rule", E + "nowrap/task.py",
     'checks = dict(g["checks"], scan_time_within_factor=g["scan_s"] <= TOL * best["scan_s"] + 1e-9)', 'checks = dict(g["checks"], scan_time_within_factor=True)'),
    ("M23", "scoring", "NoWrap: the image is taken to be centred on the centre (the half pixel forgotten)", E + "nowrap/task.py",
     "lo, hi = -fov / 2 - pixel / 2, fov / 2 - pixel / 2 ", "lo, hi = -fov / 2, fov / 2 "),
    ("M24", "scoring", "NoWrap: the rotation of the field of view is ignored by the coverage check", E + "nowrap/task.py",
     "    v, u = in_frame(points, core.centre_yx(realised), rot)\n", "    v, u = in_frame(points, core.centre_yx(realised), 0.0)\n"),
    ("M25", "scoring", "NoWrap oracle: one line pair more than the smallest that holds the object", E + "nowrap/oracle.py",
     "n_pe = max(_even_up(need_pe / pixel_mm), 2) + extra_lines", "n_pe = max(_even_up(need_pe / pixel_mm), 2) + extra_lines + 2"),
    ("M26", "constant", "T1Transmit: the tolerance is a factor 2 (0.10 -> 1.00)", E + "t1_transmit/task.py",
     "TOLERANCE = 0.10 ", "TOLERANCE = 1.00 "),
    ("M27", "scoring", "T1Transmit: one region within the tolerance is enough", E + "t1_transmit/task.py",
     'return F.result(checks, "; ".join(bad), error=max(miss.values()), tolerance=limit,',
     'return F.result(checks if len(bad) == len(miss) else {k: True for k in checks}, "; ".join(bad), error=max(miss.values()), tolerance=limit,'),
    ("M28", "scoring", "T1Transmit: the pass window is a difference, not a factor (ln dropped)", E + "t1_transmit/task.py",
     'truth, limit = true_t1_ms(cfg, world), math.log(1.0 + cfg["task"]["tolerance"])', 'truth, limit = true_t1_ms(cfg, world), cfg["task"]["tolerance"]'),
    ("M29", "scoring", "T1Transmit: a value in seconds is accepted (the lower end of the valid range dropped)", E + "t1_transmit/task.py",
     "if not T1_VALID_MS[0] <= v <= T1_VALID_MS[1]:", "if not 0 < v <= T1_VALID_MS[1]:"),
    ("M30", "scoring", "T1Transmit: every region is compared with the T1 of the first region", E + "t1_transmit/task.py",
     'return {r["id"]: float(truth[r["id"]]["t1_ms"]) for r in cfg["task"]["regions"]}',
     'return {r["id"]: float(truth[cfg["task"]["regions"][0]["id"]]["t1_ms"]) for r in cfg["task"]["regions"]}'),
    # ---- engine 0.2.0: the slice-selective path (M02, M03 and M13 edit the hard-pulse forms, which now serve
    # ``excitation: nonselective`` only; these are the same kind of defect where the default scan is computed)
    # engine 0.3.0: the cycle of the inversion recovery is written once (signal.ir_events, mutant M03) for hard and
    # selective pulses; the defect of the selective path that is left to plant is the inversion efficiency
    ("M31", "physics", "slice-selective inversion recovery takes the inversion as complete", S + "kspace.py",
     'return S.ir_events(theta, rp["tr_ms"], rp["te_ms"], rp["ti_ms"], inv_eff, 1 - 2 * echo), np.sin(theta) * echo',
     'return S.ir_events(theta, rp["tr_ms"], rp["te_ms"], rp["ti_ms"], 1.0, 1 - 2 * echo), np.sin(theta) * echo'),
    ("M32", "constant", "refocusing slice as thin as the excitation slice (1.0 for 1.5)", S + "rf.py",
     "REFOCUS_WIDTH = 1.5 ", "REFOCUS_WIDTH = 1.0 "),
    ("M33", "physics", "slice-selective spin echo: the refocusing pulse does not touch Mz", S + "kspace.py",
     'return S.se_events(theta, rp["tr_ms"], rp["te_ms"], 1 - 2 * echo), np.sin(theta) * echo',
     'return S.se_events(theta, rp["tr_ms"], rp["te_ms"], 1 - 0 * echo), np.sin(theta) * echo'),
    ("M34", "physics", "partial Fourier: the lines without a conjugate partner weigh 1 (zero filling under a homodyne reconstruction)", S + "kspace.py",
     "np.where(order > n, 2.0, np.where(order >= -n, 1.0, 0.0))", "np.where(order > n, 1.0, np.where(order >= -n, 1.0, 0.0))"),
    ("M35", "physics", "fat saturation: every line of the fat is saturated as the main line is", S + "kspace.py",
     'sat_band(df + ppm * f0 - centre, sy["fatsat_bw_hz"])', 'sat_band(df + tissue.FAT_PPM * f0 - centre, sy["fatsat_bw_hz"])'),
    ("M36", "physics", "fat saturation: nothing recovers between the saturation pulse and the excitation", S + "signal.py",
     "return es * a, 1 - es, es * b", "return es * a, 0 * (1 - es), es * b"),
    ("M37", "physics", "slice gradient at the nominal bandwidth of the pulse (not calibrated to the width at half maximum)", S + "rf.py",
     "dphi = 2 * np.pi * tbw * fwhm_factor(tbw, window, n) * zeta / n", "dphi = 2 * np.pi * tbw * 1.0 * zeta / n"),
    ("M38", "physics", "cf_shift_hz moves the off-resonance the wrong way", S + "kspace.py",
     "self.df - key[0] + key[1] * self._yx[0]", "self.df + key[0] + key[1] * self._yx[0]"),
    ("M39", "physics", "bssfp, hard pulses: the RF phase advance counted the wrong way (bands at -phase cycle / 360 / TR)", S + "kspace.py",
     'S.bssfp(a, free - np.deg2rad(rp["phase_cycle_deg"])', 'S.bssfp(a, free + np.deg2rad(rp["phase_cycle_deg"])'),
    ("M40", "physics", "bssfp of a slice: the RF phase advance counted the wrong way", S + "kspace.py",
     'th = free - np.deg2rad(rp["phase_cycle_deg"])', 'th = free + np.deg2rad(rp["phase_cycle_deg"])'),
]
REVIEW = [m[0] for m in MUTANTS[:16]]


def _limit():
    """Address-space limit of every test process (MRGYM_AUDIT_MEM_GB, default 4): a leak fails its own run."""
    import resource
    cap, hard = int(float(os.environ.get("MRGYM_AUDIT_MEM_GB", "4")) * 2 ** 30), resource.getrlimit(resource.RLIMIT_AS)[1]
    if cap <= 0:
        return
    cap = cap if hard == resource.RLIM_INFINITY else min(cap, hard)      # never above a limit the caller already runs under
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _run_groups(src_dir: str) -> dict:
    """{group: (failed, first failing test or '')} for the tests of the tree against the package in ``src_dir``."""
    env = dict(os.environ, PYTHONPATH=src_dir, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    out = {}
    for group, files in GROUPS.items():
        present = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", *present],
                           cwd=ROOT, env=env, capture_output=True, text=True, preexec_fn=_limit)
        first = next((ln.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "") for ln in r.stdout.splitlines()
                      if ln.startswith(("FAILED ", "ERROR "))), "")
        out[group] = (r.returncode != 0, first)
    return out


def _apply(mutant, work: str) -> str:
    mid, _, _, path, old, new = mutant
    dst = os.path.join(work, mid)
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(dst, "src"), ignore=shutil.ignore_patterns("__pycache__"))
    target = os.path.join(dst, path)
    with open(target, encoding="utf-8") as f:
        text = f.read()
    if text.count(old) != 1:
        raise SystemExit(f"{mid}: the text to replace occurs {text.count(old)} times in {path} (the source moved: update MUTANTS)")
    with open(target, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))
    return os.path.join(dst, "src")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    chosen = [m for m in MUTANTS if a.only is None or m[0] in a.only]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="mrgym-hand-mutants-") as work:
        base_src = os.path.join(work, "base", "src")
        shutil.copytree(os.path.join(ROOT, "src"), base_src, ignore=shutil.ignore_patterns("__pycache__"))
        base = _run_groups(base_src)
        if any(failed for failed, _ in base.values()):
            print("the unmodified tree fails:", {g: first for g, (failed, first) in base.items() if failed})
            return 2
        srcs = {m[0]: _apply(m, work) for m in chosen}
        with concurrent.futures.ThreadPoolExecutor(max(1, min(a.workers, 2))) as pool:
            results = dict(zip(srcs, pool.map(_run_groups, srcs.values())))
    rows = []
    print(f"{'id':<5}{'kind':<12}{'scoring':<9}{'metamorphic':<13}{'golden':<8}what   (first failing test)")
    for mid, kind, what, *_ in chosen:
        r = results[mid]
        killed = any(failed for failed, _ in r.values())
        first = next((f for failed, f in r.values() if failed), "")
        rows.append(dict(id=mid, kind=kind, what=what, killed=killed, by=[g for g, (failed, _) in r.items() if failed], first_failing_test=first))
        mark = {g: "killed" if r[g][0] else "-" for g in GROUPS}
        print(f"{mid:<5}{kind:<12}{mark['scoring']:<9}{mark['metamorphic']:<13}{mark['golden']:<8}{what}   ({first or 'SURVIVED'})")
    review = [r for r in rows if r["id"] in REVIEW]
    extra = [r for r in rows if r["id"] not in REVIEW]
    summary = dict(review_killed=sum(r["killed"] for r in review), review_total=len(review),
                   extra_killed=sum(r["killed"] for r in extra), extra_total=len(extra), seconds=round(time.time() - t0, 1))
    print(f"review mutants killed {summary['review_killed']} / {summary['review_total']}; "
          f"further mutants (constants, slice-selective path) killed {summary['extra_killed']} / {summary['extra_total']}; {summary['seconds']} s")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(summary=summary, mutants=rows), f, indent=1)
    return 0 if all(r["killed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
