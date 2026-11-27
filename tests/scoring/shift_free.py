"""Scoring table of ``Scanner/ShiftFree-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), per region, on the noise-free target scan with the delivered settings:
  fat_off     the mean fat signal over the area of the region is at most 5 % of its mean water signal
  snr_floor   the region's signal-to-noise ratio is at least the stated floor (its ratio at 600 Hz per pixel)
  near_best   that ratio is at least 88 % of the highest one any setting reaches under fat_off

The expected verdicts are derived here from the hidden geometry, the hidden materials and the displacement law, with
a model of its own (no image is formed here, and nothing comes from the task module):

  displacement  a fat line that lies f below water appears p x (f - df) / bandwidth pixels along the readout with
                ``ro_polarity`` p (df: the field offset at the fat border; docs/scanner.md, "Encoding": towards positive
                positions with polarity 1). So the fat on the negative side of a region comes closer with polarity 1
                and the fat on the positive side with -1. Fat has six lines (Hamilton 2011, the table below): each
                moves its share of the fat disc by its own displacement
  overlap       the fat disc (radius 9 mm) and the region (radius 5 mm) are circles whose borders are ``gap`` apart: a
                line displaced by d covers the lens of two circles whose centres are 14 mm + gap - d apart
  fat share     (fat signal / gel signal) x sum over the lines of (share of the line) x (lens area) / (area of the
                region), with the spin-echo signal PD x (1 - 2 exp(-(TR - TE / 2) / T1) + exp(-TR / T1)) x exp(-TE / T2)
                of the gel and of this fat disc (its drawn T1)
  threshold     the main-line displacement at which this share is 5 %: ``threshold_mm``. The model has sharp borders
                and no pixel; the image has a soft border and a pixel of 1.25 mm, which carry fat across the border a
                little earlier (0.2 to 0.5 mm on the instances of this table). The verifier's own threshold must lie
                within [-0.60, +0.10] mm of the model's (``agrees_with_the_model``); rows are placed 0.75 mm inside
                the model's threshold (off) or 1 mm beyond it (on)
  near_best     signal-to-noise ratio is proportional to 1 / sqrt(bandwidth) at a fixed echo time, so 0.75 mm inside a
                threshold of 3.9 mm or more (the smallest the drawn gaps give) is within 0.88 of the best; twice that
                bandwidth, or the bandwidth the narrower side needs, is not
  snr_floor     fails for a bandwidth above 600 Hz per pixel and holds below
"""
import math

from scoring import Row

ENV_ID = "Scanner/ShiftFree-v0"
INSTANCES = {"L2": dict(level=2, seed=0), "L3": dict(level=3, seed=0)}
GAMMA_HZ_PER_T, PIXEL_MM = 42.577478518e6, 240.0 / 192
FAT_LINES = ((3.80, 0.087), (3.40, 0.693), (2.60, 0.128), (1.94, 0.004), (0.39, 0.039), (-0.60, 0.048))   # ppm below water, share
MAIN_PPM = 3.40
FAT = dict(pd=0.95, t2_ms=68.0)                    # subcutaneous fat at 3 T (de Bazelaire 2004); the T1 of each disc is drawn (hidden truth)
TR_MS, TE_MS = 500.0, 30.0
REGION_MM, FAT_MM = 5.0, 9.0
COVER_MAX, NEAR_BEST = 0.05, 0.88
INSIDE_MM, BEYOND_MM = 0.75, 1.0
UNIT = {"x-": (0.0, -1.0), "x+": (0.0, 1.0), "y-": (-1.0, 0.0), "y+": (1.0, 0.0)}


def sites(case):
    return {s["id"]: s for s in case.cfg["truth"]["sites"]}


def setting_for(side):
    """(pe_dir, ro_polarity) with which the fat on ``side`` moves towards the region: the readout runs along the axis
    of the side; polarity 1 moves fat towards positive positions, so the fat on the negative side approaches."""
    return ("y" if side[0] == "x" else "x"), (1 if side[1] == "-" else -1)


def field_offset_hz(case, site, side):
    """Off-resonance at the fat border on ``side`` of the region: the field map there plus the centre-frequency error."""
    w = case.world
    reach = site["radius_mm"] + site["gap_mm"][side]
    y, x = site["y_mm"] + UNIT[side][0] * reach, site["x_mm"] + UNIT[side][1] * reach
    d = w["extent_mm"] / w["n"]
    i, j = int(round(y / d)) + w["n"] // 2, int(round(x / d)) + w["n"] // 2
    return float(w["fields"]["b0_hz"][i, j] + w["system"]["cf_offset_hz"])


def line_mm(case, site, side, bandwidth, ppm=MAIN_PPM):
    """Displacement towards the region of the fat line ``ppm`` below water, for the fat on ``side``."""
    return (ppm * 1e-6 * GAMMA_HZ_PER_T * case.world["b0_t"] - field_offset_hz(case, site, side)) / bandwidth * PIXEL_MM


def lens_mm2(penetration):
    """Area that two circles of radius 9 and 5 mm share when the smaller one reaches ``penetration`` mm into the other."""
    if penetration <= 0:
        return 0.0
    c = FAT_MM + REGION_MM - penetration
    if c <= FAT_MM - REGION_MM:
        return math.pi * REGION_MM ** 2
    a = REGION_MM ** 2 * math.acos((c * c + REGION_MM ** 2 - FAT_MM ** 2) / (2 * c * REGION_MM))
    b = FAT_MM ** 2 * math.acos((c * c + FAT_MM ** 2 - REGION_MM ** 2) / (2 * c * FAT_MM))
    return a + b - 0.5 * math.sqrt((-c + REGION_MM + FAT_MM) * (c + REGION_MM - FAT_MM) * (c - REGION_MM + FAT_MM) * (c + REGION_MM + FAT_MM))


def spin_echo(t):
    return t["pd"] * (1 - 2 * math.exp(-(TR_MS - TE_MS / 2) / t["t1_ms"]) + math.exp(-TR_MS / t["t1_ms"])) * math.exp(-TE_MS / t["t2_ms"])


def fat_share(case, site, side, bandwidth):
    """The model's fat share of the region when the fat on ``side`` is the one that approaches."""
    contrast = spin_echo(dict(FAT, t1_ms=site["fat_t1_ms"][side])) / spin_echo(case.world["tissues"]["body"])
    lenses = sum(share * lens_mm2(line_mm(case, site, side, bandwidth, ppm) - site["gap_mm"][side]) for ppm, share in FAT_LINES)
    return contrast * lenses / sum(share for _, share in FAT_LINES) / (math.pi * REGION_MM ** 2)


def threshold_mm(case, site, side):
    """Main-line displacement at which the model's fat share is 5 % (bisection on the bandwidth)."""
    lo, hi = 20.0, 3000.0                                                # Hz per pixel: on the region, off the region
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if fat_share(case, site, side, mid) > COVER_MAX else (lo, mid)
    return line_mm(case, site, side, hi)


def bandwidth_for(case, site, side, displacement_mm):
    return line_mm(case, site, side, 1.0) / displacement_mm


def scan(case, site, side, offset_mm=-INSIDE_MM, factor=1.0):
    """The setting that moves the fat on ``side`` towards the region, its main line ``offset_mm`` from the threshold."""
    pe, pol = setting_for(side)
    bw = factor * bandwidth_for(case, site, side, threshold_mm(case, site, side) + offset_mm)
    full = dict(bandwidth_hz_per_px=round(float(bw), 2), ro_polarity=pol, pe_dir=pe)
    return {k: full[k] for k in case.cfg["task"]["free"]}


def widest(site):
    return max(site["gap_mm"], key=site["gap_mm"].get)


def narrowest(site):
    return min(site["gap_mm"], key=site["gap_mm"].get)


def right(case, **change):
    """Every region with the fat of its widest gap 0.75 mm inside the model's threshold; ``R1=...`` replaces one scan."""
    s = sites(case)
    return dict(scans={i: change.get(i) or scan(case, s[i], widest(s[i])) for i in s})


def changed(region, how):
    return lambda case: right(case, **{region: how(case, sites(case)[region])})


def constant(bandwidth, **more):
    def answer(case):
        full = dict(dict(bandwidth_hz_per_px=bandwidth, ro_polarity=1, pe_dir="y"), **more)
        return dict(scans={i: {k: full[k] for k in case.cfg["task"]["free"]} for i in sites(case)})
    return answer


def agrees_with_the_model(case, score):
    """Every region: the reported numbers agree with the law and the model, and the checks are the three rules."""
    assert set(score["regions"]) == set(sites(case))
    for i, row in score["regions"].items():
        s, site = score["answer"]["scans"][i], sites(case)[i]
        nominal = MAIN_PPM * 1e-6 * GAMMA_HZ_PER_T * case.world["b0_t"] / s["bandwidth_hz_per_px"] * PIXEL_MM
        assert abs(row["displacement_mm"] - nominal) < 1e-9             # what the header's water-fat shift says, in mm
        assert score["checks"][f"fat_off_{i}"] is (row["cover"] <= COVER_MAX)
        assert score["checks"][f"snr_floor_{i}"] is (row["snr"] >= case.cfg["task"]["snr_floor"][i])
        assert score["checks"][f"near_best_{i}"] is (row["snr"] >= NEAR_BEST * row["snr_best"])
        # the verifier's fat share against the model's, for the side this scan moves towards the region
        side = next(k for k in site["gap_mm"] if setting_for(k) == (s.get("pe_dir", "y"), s["ro_polarity"]))
        model = fat_share(case, site, side, s["bandwidth_hz_per_px"])
        assert abs(row["cover"] - model) <= 0.035 + 0.25 * model, (i, row["cover"], model)
        # the best setting moves the fat of the widest gap, to the model's threshold or a little short of it
        best = row["best"]
        pe, pol = setting_for(widest(site))
        assert (best["ro_polarity"], best["pe_dir"]) == (pol, pe if "pe_dir" in case.cfg["task"]["free"] else "y")
        short = line_mm(case, site, widest(site), best["bandwidth_hz_per_px"]) - threshold_mm(case, site, widest(site))
        assert -0.60 <= short <= 0.10, (i, short)
        # noise grows with the square root of the bandwidth; the water signal of the region changes by the ripples of the
        # gel's own borders only (they move with the readout direction and the bandwidth: under 3 %)
        assert abs(row["snr"] / row["snr_best"] - math.sqrt(best["bandwidth_hz_per_px"] / s["bandwidth_hz_per_px"])) < 0.03
    assert score["error"] == max(r["cover"] for r in score["regions"].values())


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "regions" not in score


def without(region):
    def answer(case):
        a = right(case)
        del a["scans"][region]
        return a
    return answer


def with_key(region, **more):
    return lambda case: right(case, **{region: dict(right(case)["scans"][region], **more)})


ROWS = [
    # ---- level 2: fat on both sides along x; bandwidth and readout direction
    Row("L2", "the fat of the widest gap 0.75 mm inside the threshold", right, passed=True, expect=agrees_with_the_model),
    Row("L2", "one region 1 mm beyond the threshold", changed("R1", lambda c, s: scan(c, s, widest(s), BEYOND_MM)),
        passed=False, failing=("fat_off_R1",), reason="R1: fat is", expect=agrees_with_the_model),
    Row("L2", "one region at twice the bandwidth it needs", changed("R2", lambda c, s: scan(c, s, widest(s), factor=2.0)),
        passed=False, failing=("near_best_R2",), reason="R2: signal-to-noise ratio", expect=agrees_with_the_model),
    Row("L2", "one region towards its narrow side, with the bandwidth that side needs", changed("R3", lambda c, s: scan(c, s, narrowest(s))),
        passed=False, failing=("near_best_R3",), reason="R3: signal-to-noise ratio", expect=agrees_with_the_model),
    Row("L2", "the right bandwidth in the wrong direction", changed("R1", lambda c, s: dict(scan(c, s, widest(s)), ro_polarity=-setting_for(widest(s))[1])),
        passed=False, failing=("fat_off_R1",), reason="R1: fat is", expect=agrees_with_the_model),
    Row("L2", "1500 Hz per pixel everywhere: no fat, no signal-to-noise ratio", constant(1500.0), passed=False,
        failing=tuple(f"{c}_{r}" for r in ("R1", "R2", "R3") for c in ("snr_floor", "near_best")), reason="is below the floor",
        expect=agrees_with_the_model),
    Row("L2", "integers are numbers", lambda c: dict(scans={i: dict(s, bandwidth_hz_per_px=int(math.ceil(s["bandwidth_hz_per_px"])))
                                                        for i, s in right(c)["scans"].items()}), passed=True),
    # ---- level 3: fat on four sides; the phase-encode axis is delivered too
    Row("L3", "the fat of the widest of four gaps 0.75 mm inside the threshold", right, passed=True, expect=agrees_with_the_model),
    Row("L3", "one region with the readout along the other axis, with the bandwidth its wider side there needs",
        changed("R2", lambda c, s: scan(c, s, max((k for k in s["gap_mm"] if k[0] != widest(s)[0]), key=s["gap_mm"].get))),
        passed=False, failing=("near_best_R2",), reason="R2: signal-to-noise ratio", expect=agrees_with_the_model),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L2", "the right bandwidths in kHz per pixel", lambda c: dict(scans={i: dict(s, bandwidth_hz_per_px=s["bandwidth_hz_per_px"] / 1000.0)
                                                                         for i, s in right(c)["scans"].items()}),
        refused=True, reason="the console rejects the scan of R1: bandwidth_hz_per_px=", expect=not_evaluated),
    Row("L2", "a bandwidth the gradients cannot play at this matrix", changed("R2", lambda c, s: dict(scan(c, s, widest(s)), bandwidth_hz_per_px=2600.0)),
        refused=True, reason="the console rejects the scan of R2: readout gradient is"),
    Row("L2", "a region is missing", without("R3"), refused=True, reason='the deliverable is {"scans": {"R1": {"bandwidth_hz_per_px": ..., "ro_polarity": ...}'),
    Row("L2", "an axis where the task fixes it", with_key("R1", pe_dir="x"), refused=True, reason="the scan of R1 needs exactly bandwidth_hz_per_px, ro_polarity"),
    Row("L3", "no axis where the task leaves it free", lambda c: dict(scans={i: {k: v for k, v in s.items() if k != "pe_dir"} for i, s in right(c)["scans"].items()}),
        refused=True, reason="the scan of R1 needs exactly bandwidth_hz_per_px, ro_polarity, pe_dir"),
    Row("L2", "a readout direction that does not exist", with_key("R2", ro_polarity=0), refused=True, reason="the console rejects the scan of R2: ro_polarity must be one of"),
    Row("L2", "not an object", [120.0, 1], refused=True, reason="the deliverable is"),
]
