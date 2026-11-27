"""Scoring table of ``Scanner/GhostAside-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), per tube, on the noise-free target scan with the delivered axis and repetition time,
against the same scan of the object at rest:
  ghost_off   in each of the tube's two regions the mean ghost signal is at most 2 % of the region's mean signal
  fastest     the scan time is at most 10 % above that of the fastest scan that satisfies ghost_off

The expected verdicts are derived here from the hidden rates and the ghost law, not from the task module:

  law        a tube whose signal pulsates at the rate f leaves two copies of itself along the phase-encode axis, at
             g = field of view x dist(f x TR) on either side (one average; dist = distance from the nearest whole number)
  geometry   a copy is a disc of 5 mm radius, a region a disc of 12 mm radius whose centre lies D from the tube: the
             copy is geometrically clear of the region at g >= D + 17 mm and lies well inside it for |g - D| <= 12 mm.
             Rule ghost_off is reached a little before the copy is clear (the 2 % allow a sliver of it, more of a
             weak ghost than of a strong one): between g = D + 13.8 and D + 16.2 mm. A ghost only reaches the region on its own axis: region A (the tube's column)
             with pe_dir y, region B (its row) with pe_dir x
  time       scan time = 3 s + TR x (1 dummy cycle + 128 lines) (docs/scanner.md, "Cost")
  fastest    for a slow tube the ghost moves outward as TR grows, so the first clear TR on an axis is
             (D + 13.8 to 16.2 mm) / (field of view x f), and the nearer region's axis is the faster one. A scan at
             g = D + 16.5 mm on that axis is clear and at most 8 % slower than the fastest. The other region lies at
             least 6 mm further away: its axis costs 15 % or more. A tube that is still, or whose ghosts lie far
             beyond its regions at the shortest TR, is served fastest at the shortest TR
"""
import math

from scoring import Row

ENV_ID = "Scanner/GhostAside-v0"
INSTANCES = {"L2": dict(level=2, seed=0), "L3": dict(level=3, seed=0)}
FOV_MM, LINES, OVERHEAD_S = 240.0, 128, 3.0
TR_LO, TR_HI = 300.0, 1000.0
GHOST_MAX, TIME_TOL = 0.02, 1.10
CLEAR_MM = 16.5                     # a copy this far beyond a region's centre is clear of it under the rule


def tubes(case):
    return {u["id"]: u for u in case.cfg["truth"]["tubes"]}


def ghost_mm(rate, tr_ms):
    u = rate * tr_ms * 1e-3
    return FOV_MM * abs(u - round(u))


def scan_s(tr_ms):
    return OVERHEAD_S + tr_ms * 1e-3 * (1 + LINES)


def distance(tube, pe):
    r = tube["regions"][0 if pe == "y" else 1]                           # A: the column (pe_dir y); B: the row (pe_dir x)
    assert r["id"].endswith("A" if pe == "y" else "B")
    return math.hypot(r["y_mm"] - tube["y_mm"], r["x_mm"] - tube["x_mm"])


def nearer(tube):
    return min(("y", "x"), key=lambda pe: distance(tube, pe))


def clear_tr(tube, pe, beyond=CLEAR_MM):
    """Repetition time at which the law puts the ghost ``beyond`` mm past the centre of the region on axis ``pe``."""
    return 1e3 * (distance(tube, pe) + beyond) / (FOV_MM * tube["rate_hz"])


def ghost_lies_on_a_region_at_the_shortest_tr(tube):
    return tube["rate_hz"] is not None and any(abs(ghost_mm(tube["rate_hz"], TR_LO) - distance(tube, pe)) <= 12.0 for pe in ("y", "x"))


def right_scan(tube):
    if tube["rate_hz"] is None or not ghost_lies_on_a_region_at_the_shortest_tr(tube):
        return dict(pe_dir="y", tr_ms=TR_LO)
    pe = nearer(tube)
    return dict(pe_dir=pe, tr_ms=round(clear_tr(tube, pe), 1))


def right(case, **change):
    return dict(scans={i: change.get(i) or right_scan(u) for i, u in tubes(case).items()})


def slow(case):
    """Ids of the tubes whose ghost lies on a region at the shortest repetition time, the fastest pulsation first."""
    return sorted((i for i, u in tubes(case).items() if ghost_lies_on_a_region_at_the_shortest_tr(u)), key=lambda i: -tubes(case)[i]["rate_hz"])


def other_tube(case):
    """The tube of level 3 that is still or fast."""
    return next(i for i, u in tubes(case).items() if not ghost_lies_on_a_region_at_the_shortest_tr(u))


def agrees_with_the_law(case, score):
    """Per tube: the ghost share is where the law puts it, the fastest scan is the nearer region's clearing time, the
    scan times are the console's arithmetic, and the checks are the two rules."""
    for i, u in tubes(case).items():
        row, s = score["sites"][i], score["answer"]["scans"][i]
        assert abs(row["scan_s"] - scan_s(s["tr_ms"])) < 1e-6
        assert score["checks"][f"ghost_off_{i}"] is (row["ghost"] <= GHOST_MAX)
        assert score["checks"][f"fastest_{i}"] is (row["scan_s"] <= TIME_TOL * row["scan_s_fastest"])
        assert row["ghost"] == max(row["by_region"].values()) and set(row["by_region"]) == {r["id"] for r in u["regions"]}
        if not ghost_lies_on_a_region_at_the_shortest_tr(u):
            assert row["fastest"]["tr_ms"] == TR_LO and abs(row["scan_s_fastest"] - scan_s(TR_LO)) < 1e-6
        else:
            pe = nearer(u)
            assert row["fastest"]["pe_dir"] == pe
            beyond = ghost_mm(u["rate_hz"], row["fastest"]["tr_ms"]) - distance(u, pe)
            assert 13.8 < beyond < 16.2, (i, beyond)
        g = None if u["rate_hz"] is None else ghost_mm(u["rate_hz"], s["tr_ms"])
        d = distance(u, s["pe_dir"])
        if g is None or abs(g - d) >= 17.5:
            assert row["by_region"][u["regions"][0 if s["pe_dir"] == "y" else 1]["id"]] <= 0.005     # clear of the region on its axis
        elif abs(g - d) <= 12.0:
            assert row["ghost"] > 0.08                                   # well inside: several times the limit
        other = u["regions"][1 if s["pe_dir"] == "y" else 0]["id"]
        assert row["by_region"][other] <= 0.005                          # the region of the other axis is never reached
    assert score["error"] == max(r["ghost"] for r in score["sites"].values())


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "sites" not in score


def changed(tube, how):
    return lambda case: right(case, **{tube: how(tubes(case)[tube])})


def on_the_other_axis(u):
    pe = "x" if nearer(u) == "y" else "y"
    return dict(pe_dir=pe, tr_ms=round(clear_tr(u, pe), 1))


def the_fast_one_slowly(case):
    assert other_tube(case) == "T2" and tubes(case)["T2"]["kind"] == "fast"     # the instance of the table (level 3, seed 0)
    return right(case, T2=dict(pe_dir="y", tr_ms=450.0))


ROWS = [
    # ---- level 2: three slow tubes; at the shortest TR every ghost lies on a region with either axis
    Row("L2", "every ghost 16.5 mm past the centre of the nearer region", right, passed=True, expect=agrees_with_the_law),
    Row("L2", "the shortest repetition time for every tube", lambda c: dict(scans={i: dict(pe_dir="y", tr_ms=TR_LO) for i in tubes(c)}),
        passed=False, failing=("ghost_off_T1", "ghost_off_T2", "ghost_off_T3"), reason="T1: the ghost in T1A is", expect=agrees_with_the_law),
    Row("L2", "the shortest repetition time with the other axis: the ghost lies on the other region",
        lambda c: dict(scans={i: dict(pe_dir="x", tr_ms=TR_LO) for i in tubes(c)}),
        passed=False, failing=("ghost_off_T1", "ghost_off_T2", "ghost_off_T3"), reason="T1: the ghost in T1B is", expect=agrees_with_the_law),
    Row("L2", "one tube at the shortest repetition time", changed("T1", lambda u: dict(pe_dir=nearer(u), tr_ms=TR_LO)),
        passed=False, failing=("ghost_off_T1",), reason="T1: the ghost in T1", expect=agrees_with_the_law),
    Row("L2", "one tube cleared on the axis of its further region: 15 % slower or more", changed("T2", on_the_other_axis),
        passed=False, failing=("fastest_T2",), reason="T2: ", expect=agrees_with_the_law),
    Row("L2", "one tube 20 % slower than it needs", changed("T3", lambda u: dict(pe_dir=nearer(u), tr_ms=round(1.2 * clear_tr(u, nearer(u)), 1))),
        passed=False, failing=("fastest_T3",), reason="above the fastest scan without a ghost", expect=agrees_with_the_law),
    Row("L2", "integers are numbers", lambda c: dict(scans={i: dict(s, tr_ms=int(math.ceil(s["tr_ms"]))) for i, s in right(c)["scans"].items()}), passed=True),
    # ---- level 3: one tube is still or fast
    Row("L3", "the slow tubes cleared, the other one at the shortest repetition time", right, passed=True, expect=agrees_with_the_law),
    Row("L3", "the tube without a ghost on its regions scanned slowly", the_fast_one_slowly,
        passed=False, failing=("fastest_T2",), reason="T2: 61.0 s is 46 % above the fastest scan without a ghost", expect=agrees_with_the_law),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L2", "the right repetition times in seconds", lambda c: dict(scans={i: dict(s, tr_ms=s["tr_ms"] / 1000.0) for i, s in right(c)["scans"].items()}),
        refused=True, reason="the console rejects the scan of T1: tr_ms=", expect=not_evaluated),
    Row("L2", "a repetition time below the window", lambda c: right(c, T2=dict(pe_dir="y", tr_ms=250.0)), refused=True,
        reason="tr_ms of T2 is 250: the examination allows 300 to 1000 ms"),
    Row("L2", "a repetition time above the window", lambda c: right(c, T3=dict(pe_dir="x", tr_ms=1200.0)), refused=True,
        reason="tr_ms of T3 is 1200: the examination allows 300 to 1000 ms"),
    Row("L2", "a tube is missing", lambda c: dict(scans={i: s for i, s in right(c)["scans"].items() if i != "T2"}), refused=True,
        reason='the deliverable is {"scans": {"T1": {"pe_dir": ..., "tr_ms": ...}'),
    Row("L2", "averages are not delivered", lambda c: right(c, T1=dict(right(c)["scans"]["T1"], averages=2)), refused=True,
        reason="the scan of T1 needs exactly pe_dir and tr_ms"),
    Row("L2", "an axis that does not exist", lambda c: right(c, T1=dict(pe_dir="z", tr_ms=500.0)), refused=True,
        reason="the console rejects the scan of T1: pe_dir must be one of"),
    Row("L2", "not an object", "T1 y 500", refused=True, reason="the deliverable is"),
]
