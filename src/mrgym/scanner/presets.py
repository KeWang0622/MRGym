"""Named starting protocols, computed from the tissue table of this package (``tissue.table``) for the field strength
of the session. ``scanner_info(topic="presets")`` prints them.

A preset is a starting point a console would offer: its tissue-dependent numbers (inversion time, echo time, flip
angle, the repetition time where contrast depends on it) are computed here from the table and the stated rule, so that
a preset does on this scanner what its name says. They are computed from the NOMINAL table: the tissue of a session
is drawn round it and the system has errors, so a preset is never the answer to a task. Geometry is left at the
console defaults. The three ``acr_*`` presets are the series of the ACR phantom guidance (``mrgym.qa.SERIES``).

Rules
  t1w_se        spin echo, TE at the console minimum, TR that maximises the white / grey matter difference
  pdw_se        TE at the console minimum, TR = 3 x the longer T1 of white and grey matter
  t2w_se        the same TR, TE that maximises the white / grey matter difference
  flair         inversion recovery, TR = 2 x T1 of CSF, TI at the null of CSF for that TR
  stir          inversion recovery, TR = 3 x T1 of white matter, TI at the null of subcutaneous fat for that TR
  t1w_spgr      spoiled gradient echo at TR 20 ms with the Ernst angle of white matter
  in_phase /    spoiled gradient echo at the first echo time the console can realise at which water and the main fat
  opposed_phase peak are in phase (k / df) or opposed ((k + 1/2) / df), df = 3.4 ppm x gamma x B0
The null of an inversion recovery with repetition time TR (complete inversion): TI = T1 ln(2 / (1 + exp(-TR / T1))).
"""
from __future__ import annotations

import math

from . import console, tissue

GAMMA_HZ_PER_T = 42.577478518e6


def null_ti_ms(t1_ms: float, tr_ms: float) -> float:
    """Inversion time that nulls a material of the given T1 at repetition time TR (complete inversion)."""
    return t1_ms * math.log(2.0 / (1.0 + math.exp(-tr_ms / t1_ms)))


def _best(f, lo, hi, n=2000):
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return max(xs, key=lambda x: abs(f(x)))


def presets(b0_t: float, exam: dict | None = None) -> dict:
    """{name: dict(protocol=..., rule=...)} for 1.5 T or 3 T. Times rounded to 1 ms (echo times up to 0.01 ms).
    ``exam``: the console card of the session (its default field of view sets the shortest echo time)."""
    t = tissue.table(b0_t)
    wm, gm, csf, fat = t["wm"], t["gm"], t["csf"], t["subcut_fat"]
    se = lambda c, tr, te: c["pd"] * (1 - math.exp(-tr / c["t1_ms"])) * math.exp(-te / c["t2_ms"])      # noqa: E731
    tr_t1 = round(_best(lambda tr: se(wm, tr, 0.0) - se(gm, tr, 0.0), 150.0, 3000.0))
    tr_long = round(3.0 * max(wm["t1_ms"], gm["t1_ms"]))
    te_t2 = round(_best(lambda te: se(wm, tr_long, te) - se(gm, tr_long, te), 20.0, 300.0))
    tr_flair, tr_stir = round(2.0 * csf["t1_ms"]), round(3.0 * wm["t1_ms"])
    tr_gre = 20.0
    ernst = math.degrees(math.acos(math.exp(-tr_gre / wm["t1_ms"])))
    df = abs(tissue.FAT_PPM) * 1e-6 * GAMMA_HZ_PER_T * b0_t                                       # Hz
    period = 1e3 / df
    te_min = console.plan(dict(sequence="spgr", tr_ms=tr_gre, flip_deg=round(ernst, 1)), dict(exam or {}, b0_t=b0_t))["timing"]["te_min_ms"]
    te_in = period * math.ceil(te_min / period)
    te_op = period * (math.ceil(te_min / period - 0.5) + 0.5)
    from .. import qa
    out = dict(
        t1w_se=dict(protocol=dict(sequence="se", tr_ms=float(tr_t1), te_ms="min"),
                    rule=f"TR maximises |white - grey matter| (T1 {wm['t1_ms']:g} / {gm['t1_ms']:g} ms)"),
        pdw_se=dict(protocol=dict(sequence="se", tr_ms=float(tr_long), te_ms="min"), rule="TR = 3 x the longer T1 of white and grey matter"),
        t2w_se=dict(protocol=dict(sequence="se", tr_ms=float(tr_long), te_ms=float(te_t2)),
                    rule=f"TE maximises |white - grey matter| (T2 {wm['t2_ms']:g} / {gm['t2_ms']:g} ms)"),
        flair=dict(protocol=dict(sequence="ir", tr_ms=float(tr_flair), ti_ms=float(round(null_ti_ms(csf["t1_ms"], tr_flair)))),
                   rule=f"TR = 2 x T1 of CSF ({csf['t1_ms']:g} ms); TI = its null at that TR"),
        stir=dict(protocol=dict(sequence="ir", tr_ms=float(tr_stir), ti_ms=float(round(null_ti_ms(fat["t1_ms"], tr_stir)))),
                  rule=f"TR = 3 x T1 of white matter; TI = the null of subcutaneous fat (T1 {fat['t1_ms']:g} ms) at that TR"),
        t1w_spgr=dict(protocol=dict(sequence="spgr", tr_ms=tr_gre, flip_deg=round(ernst, 1)), rule="flip angle = Ernst angle of white matter at TR 20 ms"),
        in_phase=dict(protocol=dict(sequence="spgr", tr_ms=tr_gre, flip_deg=round(ernst, 1), te_ms=math.ceil(te_in * 100) / 100),
                      rule=f"TE = k / df, df = {df:.0f} Hz (water and the main fat peak in phase)"),
        opposed_phase=dict(protocol=dict(sequence="spgr", tr_ms=tr_gre, flip_deg=round(ernst, 1), te_ms=math.ceil(te_op * 100) / 100),
                           rule=f"TE = (k + 1/2) / df, df = {df:.0f} Hz (water and the main fat peak opposed)"),
    )
    for name, series in (("acr_localizer", "localizer"), ("acr_t1", "t1"), ("acr_t2", "t2")):
        out[name] = dict(protocol=dict(qa.SERIES[series]), rule="ACR phantom guidance, Table 2 (the bandwidth is this package's choice)")
    return out
