"""Public physics of ``Scanner/RFPulse``: what the task card, the instrument manual and the design layer let anyone compute.

Nothing here reads a world. The verifier calls ``search`` with the transmit chain of the hidden scanner; a control
policy calls it with the chain it measured. Both get the same procedure: that is the "stated reference search" of the
pass rule (``task.py``), and it is stated here in full.

The reference search, ``search(spec, limits, chains)``
  Candidates are Shinnar-Le Roux excitations of the ``slr`` generator (equiripple in the flip angle, linear phase,
  constant gradient). A candidate is fixed by three numbers:
    share    (in-slice, out-of-slice): the parts of the two tolerances of the specification the filter is built to
             (``SHARES``); the rest is left for what the transmit chain does to the pulse
    tbw      the time-bandwidth product. For every share the smallest tbw (to ``TBW_STEP``) whose design meets the mask
             on the chain WITHOUT the amplifier law is found (``smallest_tbw``, ``TBW_RANGE``); candidates are that tbw
             times each factor of ``TBW_MARGINS``
    stretch  the factor (>= 1) by which the pulse is longer than the shortest duration the limits allow. A longer
             pulse has a lower amplitude and is compressed less by the amplifier. For every (share, tbw) the smallest
             stretch at which the candidate is valid is found: the first valid step of ``STRETCH_LADDER``, then a
             bisection between it and the step before (to ``STRETCH_STEP``, relative); on a chain without the
             amplifier law the stretch is 1 and the tbw is not widened
  The shortest duration (``shortest``) is the first one on the 0.01 ms raster at which the gradient, the peak B1 and
  the RF power are within their limits. The amplitude of a candidate is set so that the middle of the flip
  angles in the slice, on the first chain, is the specified angle (``matched``, ``MATCH_ROUNDS`` rounds). A candidate
  is valid when its profile meets the mask on EVERY chain given (the verifier gives one, the hidden scanner's; a
  policy gives the corners of what it measured). The result is the shortest valid candidate.

It is a search over one generator, not an optimum: a pulse of another kind (control points, non-linear phase) can be
shorter than what it returns. A design shorter than the reference passes; it is recorded, not rejected.

The floor, ``floor_ms``: no pulse can turn a spin further than the sum of its sub-pulse angles (the triangle
inequality for rotations; precession about z does not move +z), so the flip angle anywhere is at most
2 pi gamma x (largest B1 a spin can receive) x duration. With the peak limit on the nominal waveform and the
transmit scale s of the chain the largest B1 is s x limit (the amplifier only lowers it), hence
    duration >= (smallest flip angle the slice needs) / (2 pi gamma s B1_limit).
A theorem for one transmit channel, any gradient waveform, any off-resonance, no relaxation. It is loose: measured,
3 to 18 times below the reference search (docs/physics/design.md). A delivered design below it means the engine or
this formula is wrong; it never passes.
"""
from __future__ import annotations

import math

from ...design import hardware, pulses, tool
from ...design.units import HZ_PER_MT_M_MM, RAD_PER_UT_MS, RASTER_MS

SHARES = ((0.9, 0.95), (0.7, 0.9), (0.5, 0.85))
STRETCH_MAX, STRETCH_STEP = 4.0, 0.03
STRETCH_LADDER = (1.25, 1.6, 2.0, 2.5, 3.2, 4.0)
TBW_RANGE, TBW_STEP = (1.5, 20.0), 0.05
TBW_MARGINS = (1.0, 1.12)
MATCH_ROUNDS = 3
SETTLE_STEPS = 12                                    # steps of TBW_STEP the settling on built pulses may take either way
MATCH_TOLERANCE = 5e-4                               # relative error of the middle flip angle at which matching stops


def spec_of(task: dict) -> dict:
    return dict(task["design"]["specification"])


def limits_of(task: dict) -> tool.Limits:
    lim = task["design"]["limits"]
    return tool.Limits(b1_peak_ut=lim["b1_peak_ut"], g_max_mt_m=lim["g_max_mt_m"], slew_t_m_s=lim["slew_t_m_s"], b1rms_ut=lim["b1rms_ut"],
                       tr_ms=lim["tr_ms"])


def _raster_up(t_ms: float) -> float:
    return round(math.ceil(t_ms / RASTER_MS - 1e-9) * RASTER_MS, 6)


def call_of(spec: dict, amplitude: float, tbw: float, duration_ms: float, share) -> dict:
    """The ``design`` call of a candidate: the Shinnar-Le Roux pulse for the specified angle and a share of its
    tolerances, played ``amplitude`` times as strong."""
    return dict(kind="slr", flip_deg=float(spec["flip_deg"]), duration_ms=round(float(duration_ms), 2), tbw=round(float(tbw), 3),
                thickness_mm=float(spec["thickness_mm"]), tol_in_deg=round(share[0] * spec["tol_in_deg"], 3),
                max_out_deg=round(share[1] * spec["max_out_deg"], 3), amplitude=round(float(amplitude), 4))


def gradient_ms(spec: dict, limits: tool.Limits, tbw: float) -> float:
    """Shortest duration at which the slice-select gradient of a pulse with this tbw is within its limit, on the raster."""
    return _raster_up(tbw * 1e3 / (HZ_PER_MT_M_MM * spec["thickness_mm"] * limits.g_max_mt_m))


def shortest(spec: dict, limits: tool.Limits, tbw: float, share, amplitude: float, stretch: float = 1.0):
    """(pulse, call) of the candidate at ``stretch`` times its shortest duration, or None when it cannot be built. The
    peak B1 and the energy of a pulse of fixed shape scale as 1 / duration; the samples on the raster change the shape
    a little (and the first and last sample of an equiripple pulse do not shrink at all), so the duration is raised
    until the three limits hold on the built pulse."""
    t = max(gradient_ms(spec, limits, tbw), tool.PARAMETERS["duration_ms"][0])
    energy_max = None if limits.b1rms_ut is None or limits.tr_ms is None else limits.b1rms_ut ** 2 * limits.tr_ms
    for _ in range(8):
        if t > limits.duration_max_ms:
            return None
        try:
            pulse, call = tool.build(call_of(spec, amplitude, tbw, t, share), limits)
        except tool.Refusal:
            return None
        need = max(t, t * pulse.peak_b1_ut / limits.b1_peak_ut, t * pulse.energy_ut2_ms / energy_max if energy_max else 0.0)
        need = _raster_up(need * (1 - 1e-9))
        if need <= t + 1e-9:
            break
        t = need
    else:
        return None
    if stretch <= 1.0:
        return pulse, call
    t = _raster_up(stretch * t)
    if t > limits.duration_max_ms:
        return None
    try:
        pulse, call = tool.build(call_of(spec, amplitude, tbw, t, share), limits)
    except tool.Refusal:
        return None
    return (pulse, call) if tool.facts(pulse, limits)["limits"]["ok"] else None


def meets(pulse, spec: dict, chains) -> bool:
    """Whether the profile of ``pulse`` meets every requirement of the specification on every chain."""
    return all(all(tool.figures(pulse, c, spec)["checks"].values()) for c in chains)


def matched(spec: dict, limits: tool.Limits, tbw: float, share, stretch: float, chain):
    """The candidate with its amplitude set so that the middle of the flip angles in the slice (half way between the
    smallest and the largest), on ``chain``, is the specified angle: (pulse, call) or None. On a linear chain the
    amplitude is 1 / transmit scale and one round does it (the generator centres its ripple on the angle asked for);
    the amplifier law needs more rounds."""
    amplitude = 1.0 / chain.tx_scale
    lo, hi = tool.PARAMETERS["amplitude"][:2]
    out = None
    for _ in range(MATCH_ROUNDS):
        if not lo <= amplitude <= hi:
            return None
        out = shortest(spec, limits, tbw, share, amplitude, stretch)
        if out is None:
            return None
        f = tool.figures(out[0], chain, spec)
        middle = 0.5 * (f["in_min_deg"] + f["in_max_deg"])
        if abs(middle / spec["flip_deg"] - 1) < MATCH_TOLERANCE:
            break
        amplitude *= spec["flip_deg"] / middle
    return out


def filter_transition(spec: dict, limits: tool.Limits, share, tbw: float):
    """Transition width (fraction of the thickness) of the equiripple filter of a candidate, from the filter design
    alone (no pulse is built): the sample count is that of the gradient-limited duration. None when no such filter exists."""
    n = max(16, int(round(gradient_ms(spec, limits, tbw) / RASTER_MS)))
    try:
        ripples = pulses.flip_ripples(spec["flip_deg"], share[0] * spec["tol_in_deg"], share[1] * spec["max_out_deg"])
        return pulses.slr_filter(n, tbw, "saturation", spec["flip_deg"], "pm", *ripples)[1]["transition_fraction"]
    except ValueError:
        return None


def smallest_tbw(spec: dict, limits: tool.Limits, share, chains):
    """Smallest time-bandwidth product (to ``TBW_STEP``) whose candidate meets the mask on the chains taken without
    their amplifier law, at its shortest duration. None when even the largest does not.

    The transition of an equiripple filter narrows as tbw grows, so the place is first bracketed on the filter alone
    (``filter_transition`` <= the transition of the specification; a bisection that builds no pulse) and then settled
    on built pulses: up in steps while the mask is not met, down while it still is."""
    linear = [hardware.Chain(c.tx_scale, None, c.grad_delay_us, c.df_hz) for c in chains]

    def ok(tbw):
        got = matched(spec, limits, tbw, share, 1.0, linear[0])
        return got is not None and meets(got[0], spec, linear)

    def narrow(tbw):
        w = filter_transition(spec, limits, share, tbw)
        return w is not None and w <= spec["transition"]
    lo, hi = TBW_RANGE
    if narrow(lo):
        hi = lo
    elif narrow(hi):
        while hi - lo > TBW_STEP:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if narrow(mid) else (mid, hi)
    grid = lambda t: round(round(t / TBW_STEP) * TBW_STEP, 4)            # noqa: E731
    tbw = grid(math.ceil(hi / TBW_STEP - 1e-9) * TBW_STEP)
    for _ in range(SETTLE_STEPS):                                        # up while the built pulse misses the mask
        if ok(tbw):
            break
        tbw = grid(tbw + TBW_STEP)
        if tbw > TBW_RANGE[1]:
            return None
    else:
        return None
    for _ in range(SETTLE_STEPS):                                        # down while it still meets it
        if tbw - TBW_STEP < TBW_RANGE[0] or not ok(grid(tbw - TBW_STEP)):
            break
        tbw = grid(tbw - TBW_STEP)
    return tbw


_FOUND = {}                                          # what a search returned, by its arguments: it is a function of them


def search(spec: dict, limits: tool.Limits, chains, shares=SHARES, margins=TBW_MARGINS):
    """The reference search (module docstring). ``chains``: one ``hardware.Chain`` or a list; the first one is the
    chain the flip angle is matched on. Returns ``dict(pulse, call, duration_ms, tbw, share, stretch, tried)`` of the
    shortest valid candidate, or None."""
    chains = [chains] if isinstance(chains, hardware.Chain) else list(chains)
    key = repr((sorted(spec.items()), limits, chains, shares, margins))
    if key not in _FOUND:
        if len(_FOUND) >= 256:
            _FOUND.clear()
        _FOUND[key] = _search(spec, limits, chains, shares, margins)
    return None if _FOUND[key] is None else dict(_FOUND[key])


def _search(spec, limits, chains, shares, margins):
    compressed = any(c.amp_sat_ut is not None for c in chains)
    best, tried = None, 0
    for share in shares:
        base = smallest_tbw(spec, limits, share, chains)
        if base is None:
            continue
        for m in (margins if compressed else margins[:1]):
            tbw = round(base * m, 4)
            if tbw > TBW_RANGE[1]:
                continue

            def valid(stretch, tbw=tbw, share=share):
                nonlocal tried
                tried += 1
                got = matched(spec, limits, tbw, share, stretch, chains[0])
                return got if got is not None and meets(got[0], spec, chains) else None
            first = matched(spec, limits, tbw, share, 1.0, chains[0])
            if first is None:
                continue
            tried += 1
            t_short = first[0].duration_ms
            top = STRETCH_MAX if best is None else min(STRETCH_MAX, best["duration_ms"] / t_short)
            if top < 1.0:                                                # even unstretched it would be longer than the best so far
                continue
            found, stretch = (first, 1.0) if meets(first[0], spec, chains) else (None, None)
            if found is None and compressed:
                # the first valid stretch on a coarse ladder (a long pulse has a weak gradient, and an off-resonance then
                # moves its slice: validity need not return once it is lost), then a bisection below it
                lo = 1.0
                for step in STRETCH_LADDER:
                    if step > top * (1 + 1e-9):
                        break
                    got = valid(step)
                    if got is not None:
                        found, stretch = got, step
                        break
                    lo = step
                if found is not None:
                    hi = stretch
                    while hi / lo > 1.0 + STRETCH_STEP:
                        mid = math.sqrt(lo * hi)
                        got = valid(mid)
                        if got is None:
                            lo = mid
                        else:
                            hi, found, stretch = mid, got, mid
            if found is not None and (best is None or found[0].duration_ms < best["duration_ms"] - 1e-9):
                best = dict(pulse=found[0], call=found[1], duration_ms=round(found[0].duration_ms, 6), tbw=tbw, share=tuple(share),
                            stretch=round(stretch, 4))
    if best is not None:
        best["tried"] = tried
    return best


def floor_ms(spec: dict, limits: tool.Limits, tx_scale: float) -> float:
    """The hard-pulse floor of the module docstring, ms."""
    need = math.radians(spec["flip_deg"] - spec["tol_in_deg"])
    return need / (RAD_PER_UT_MS * tx_scale * limits.b1_peak_ut)
