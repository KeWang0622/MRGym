"""RF shimming: channel weights that make |B1+| uniform in a region, under limits on power and local SAR.

Given the per-channel maps of ``solver.Solver.fields`` (B1+ in T per ampere), a set of complex channel weights w
(ampere) makes B1+ = A w at the pixels of a region (A: [pixel, channel]). The objective is the magnitude least
squares error (Setsompop et al. 2008, Magn Reson Med 59:908): the phase of B1+ does not matter for excitation, so

    f(w) = mean over the region of (|A w| - t)^2 / t^2,        t the wanted |B1+| in tesla.

Constraints are quadratic forms of w: the absorbed power per metre Re(w^H R w) / 2 <= p_max (R of
``solver.power_matrix``) and local SAR at every pixel p, Re(w^H Q_p w) <= sar_max (Q of ``solver.sar_matrices``,
point SAR or the N x N pixel average). With t equal to the B1+rms of a sequence, these are time-averaged values.
Limits as restated in the open literature for IEC 60601-2-33 are in ``IEC_LIMITS_W_PER_KG``.

Two families of weights: ``kind="full"`` (magnitude and phase of every channel) and ``kind="phase"`` (the magnitudes
of the starting weights are kept up to one common scale; only the phases and the scale change: RF shimming with one
amplifier and phase shifters).

Method. Sequential least-squares programming (``scipy.optimize.minimize``, SLSQP) with analytic gradients, restarted
until it stops improving. Without constraints and for ``kind="full"`` it starts from the result of variable exchange
(the phase of the target is set to the phase of A w, then w is the linear least-squares solution; repeated; the error
never rises). Local SAR is handled by a cutting-plane loop: the pixels that violate the limit most are added to the constraint set
and the problem is solved again, until no pixel violates it. Because every constraint is a quadratic form, any w can
be made feasible by scaling it down, and the result is scaled so if a violation of rounding size is left.

Guarantee. The result is never worse than the starting weights after their best feasible scaling
(``scaled_reference``): if the optimiser ends above that value, the scaled start is returned and
``result["fallback"]`` is True. The problem is not convex; the result is a local optimum and ``result["kkt"]``
reports the first-order optimality residual at it (``kkt_residual``).
"""
from __future__ import annotations

import numpy as np
import scipy.optimize

# Limits of IEC 60601-2-33 as restated in the open literature (W/kg, averaged over 6 minutes). Not read in the
# standard itself; see docs/physics/em.md for the source of each number.
IEC_LIMITS_W_PER_KG = dict(
    normal=dict(whole_body=2.0, head=3.2, local_head_trunk_10g=10.0, local_extremities_10g=20.0),
    first_level=dict(whole_body=4.0, head=3.2, local_head_trunk_10g=20.0, local_extremities_10g=40.0),
)


def system_matrix(b1p, mask):
    """A [pixel, channel]: the per-channel B1+ maps [channel, y, x] at the pixels of ``mask``."""
    return np.asarray(b1p)[:, mask].T.copy()


def objective(a, w, target):
    """f(w): mean of (|A w| - t)^2 / t^2 (the squared relative rms error of the magnitude)."""
    return float(np.mean((np.abs(a @ w) - target) ** 2) / target ** 2)


def uniformity(a, w):
    """Figures of |A w|: dict(mean, cv = standard deviation / mean, min_over_mean, max_over_mean)."""
    m = np.abs(a @ np.asarray(w, complex))
    return dict(mean=float(m.mean()), cv=float(m.std() / m.mean()), min_over_mean=float(m.min() / m.mean()),
                max_over_mean=float(m.max() / m.mean()))


def quad(q, w):
    """Re(w^H Q w) for one matrix [c, c] or a stack [p, c, c]."""
    w = np.asarray(w, complex)
    return np.real(np.einsum("i,...ij,j->...", np.conj(w), q, w))


def power(r, w):
    """Absorbed power per metre for the weights w: Re(w^H R w) / 2."""
    return float(quad(r, w)) / 2.0


def peak_sar(q, w):
    """Largest local SAR over the pixels of the stack Q, and its index."""
    s = quad(q, w)
    k = int(np.argmax(s))
    return float(s[k]), k


def scaled_reference(a, w0, target, r=None, p_max=None, q=None, sar_max=None):
    """The best feasible real scaling of the weights w0: the scale that minimises f, lowered to the largest scale
    the limits allow. f is a convex parabola in the scale, so this is the optimum along the ray."""
    w0 = np.asarray(w0, complex)
    m = np.abs(a @ w0)
    scale = target * m.sum() / np.sum(m * m)
    if r is not None and p_max is not None:
        scale = min(scale, np.sqrt(p_max / max(power(r, w0), 1e-300)))
    if q is not None and sar_max is not None:
        scale = min(scale, np.sqrt(sar_max / max(peak_sar(q, w0)[0], 1e-300)))
    return scale * w0


def _f_grad(a, w, target):
    """f and its gradient with respect to conj(w) (df = 2 Re(g^H dw))."""
    b = a @ w
    mag = np.maximum(np.abs(b), 1e-300)
    f = np.mean((mag - target) ** 2) / target ** 2
    g = a.conj().T @ ((1.0 - target / mag) * b) / (len(b) * target ** 2)
    return float(f), g


def variable_exchange(a, target, w0, n_iter=200, tol=1e-10):
    """Unconstrained magnitude least squares by variable exchange, from the start w0. Returns (w, f)."""
    w = np.asarray(w0, complex)
    pinv = np.linalg.pinv(a)
    f = objective(a, w, target)
    for _ in range(n_iter):
        w_new = pinv @ (target * np.exp(1j * np.angle(a @ w)))
        f_new = objective(a, w_new, target)
        if f_new > f:
            break
        done = f - f_new < tol * max(f, 1e-30)
        w, f = w_new, f_new
        if done:
            break
    return w, f


def solve(a, target, w0, r=None, p_max=None, q=None, sar_max=None, kind="full", max_rounds=30, tol=1e-4, starts=()):
    """Channel weights that minimise f(w) under the limits.

    ``a``: [pixel, channel] in T/A; ``target``: wanted |B1+| in T; ``w0``: starting weights in A (for example the
    birdcage-like mode, ``coils.mode_weights``). ``r``, ``p_max``: power matrix and limit in W/m. ``q``, ``sar_max``:
    local-SAR matrices [pixel, channel, channel] and limit in W/kg. ``kind``: "full" or "phase".

    ``starts``: further starting vectors (``random_starts``); the best result is returned, and ``start_f``,
    ``improvement`` and the guarantee refer to ``w0``.

    Returns dict(w, f, cv, mean, min_over_mean, max_over_mean, power, peak_sar, start_f, improvement, fallback,
    rounds, active, kkt, feasible, raw_f, raw_excess). ``raw_f`` and ``raw_excess`` describe what the optimiser
    itself delivered, before the final scaling and the fallback: its objective, and by how much it exceeded a limit
    (relative; 0 when it kept them all)."""
    if len(starts):
        runs = [solve(a, target, w, r, p_max, q, sar_max, kind, max_rounds, tol) for w in [w0, *starts]]
        best = min(runs, key=lambda res: res["f"])
        return dict(best, start_f=runs[0]["start_f"], improvement=runs[0]["start_f"] - best["f"])
    w0 = np.asarray(w0, complex)
    nc = len(w0)
    if kind not in ("full", "phase"):
        raise ValueError("kind must be 'full' or 'phase'")
    has_p = r is not None and p_max is not None
    has_s = q is not None and sar_max is not None
    ref = scaled_reference(a, w0, target, r if has_p else None, p_max, q if has_s else None, sar_max)
    f_ref = objective(a, ref, target)
    # dimensionless problem: weights in units of the reference scale, target 1
    unit = float(np.linalg.norm(ref) / np.sqrt(nc)) or 1.0
    a_s = a * (unit / target)
    r_s = None if not has_p else r * (unit ** 2 / (2.0 * p_max))
    q_s = None if not has_s else q * (unit ** 2 / sar_max)
    mags = np.abs(w0) / max(np.abs(w0).max(), 1e-300)

    if kind == "full":
        def to_w(x):
            return x[:nc] + 1j * x[nc:]

        def back(x, g):                                # gradient with respect to conj(w) -> gradient with respect to x
            return 2.0 * np.concatenate([g.real, g.imag])
        x0 = np.concatenate([ref.real, ref.imag]) / unit
    else:
        # variables: the phases of channels 1.. relative to their start (channel 0 keeps its phase: a common phase
        # changes nothing) and the common scale
        phase0 = np.angle(np.where(mags > 0, ref, 1.0))

        def to_w(x):
            return x[nc - 1] * mags * np.exp(1j * (phase0 + np.concatenate([[0.0], x[:nc - 1]])))

        def back(x, g):                                # dw/dtheta_c = i w_c, dw/dscale = w / scale
            w = to_w(x)
            return 2.0 * np.concatenate([np.real(np.conj(g) * 1j * w)[1:], [np.real(np.vdot(g, w)) / x[nc - 1]]])
        x0 = np.concatenate([np.zeros(nc - 1), [np.abs(ref).max() / unit]])

    def fun(x):
        f, g = _f_grad(a_s, to_w(x), 1.0)
        return f, back(x, g)

    def con(mat):
        def val(x):
            w = to_w(x)
            return 1.0 - float(np.real(np.conj(w) @ mat @ w))

        def jac(x):
            return -back(x, mat @ to_w(x))
        return dict(type="ineq", fun=val, jac=jac)

    def descend(x, cons):
        """SLSQP from x, restarted from its own result until that no longer lowers the objective (at most 4 runs).
        A start that breaks a limit just added has a lower objective than any allowed point, so the first allowed
        result is taken as it is."""
        best = fun(x)[0] if all(c["fun"](x) >= -1e-7 for c in cons) else np.inf
        for _ in range(4):
            res = scipy.optimize.minimize(fun, x, jac=True, method="SLSQP", constraints=cons, options=dict(maxiter=500, ftol=1e-14))
            if not (np.all(np.isfinite(res.x)) and all(c["fun"](res.x) >= -1e-7 for c in cons)):
                break
            gain = best - res.fun
            if res.fun <= best:
                x, best = res.x, res.fun
            if gain < 1e-13:
                break
        return x

    active = []
    x = x0
    if kind == "full" and not has_p and not has_s:
        w_s, _ = variable_exchange(a_s, 1.0, to_w(x0))
        x = np.concatenate([w_s.real, w_s.imag])
    if has_s:
        active = [peak_sar(q_s, to_w(x0))[1]]
    rounds = 0
    for rounds in range(1, max_rounds + 1):                   # noqa: B007
        x = descend(x, ([con(r_s)] if has_p else []) + [con(q_s[k]) for k in active])
        if not has_s:
            break
        s = quad(q_s, to_w(x))
        worst = [int(k) for k in np.argsort(s)[::-1][:4] if s[k] > 1.0 + tol and int(k) not in active]
        if not worst:
            break
        active += worst
    w = to_w(x) * unit
    # scale down if a limit is exceeded by what the tolerances left
    over = 1.0
    if has_p:
        over = max(over, power(r, w) / p_max)
    if has_s:
        over = max(over, peak_sar(q, w)[0] / sar_max)
    raw_f = objective(a, w, target)
    w = w / np.sqrt(over)
    f = objective(a, w, target)
    fallback = not (f <= f_ref)
    if fallback:
        w, f = ref, f_ref
    out = dict(uniformity(a, w), w=w, f=f, start_f=f_ref, improvement=f_ref - f, fallback=bool(fallback), rounds=int(rounds),
               raw_f=raw_f, raw_excess=over - 1.0,
               power=power(r, w) if r is not None else None, peak_sar=peak_sar(q, w)[0] if q is not None else None)
    out["feasible"] = bool((not has_p or out["power"] <= p_max * (1 + 1e-9)) and (not has_s or out["peak_sar"] <= sar_max * (1 + 1e-9)))
    out["active"] = dict(power=bool(has_p and out["power"] >= p_max * (1 - 1e-3)),
                         sar_pixels=int(np.sum(quad(q, w) >= sar_max * (1 - 1e-3))) if has_s else 0)
    out["kkt"] = kkt_residual(a, w, target, r if has_p else None, p_max, q if has_s else None, sar_max, kind)
    return out


def kkt_residual(a, w, target, r=None, p_max=None, q=None, sar_max=None, kind="full", near=1e-3):
    """First-order optimality at w: the smallest |grad f + sum_i mu_i grad c_i| over multipliers mu_i >= 0 of the
    constraints that are active (within ``near`` of their limit).

    The gradients are taken with respect to relative changes of the variables of ``kind``: real and imaginary parts
    in units of the rms weight, or phases in rad and the common scale in units of itself. f is dimensionless (a
    squared relative error), so the number is the change of f that a first-order step of relative size 1 could still
    buy: of the order of f itself at a start, 0 at a stationary point of the constrained problem."""
    w = np.asarray(w, complex)
    rms = float(np.linalg.norm(w) / np.sqrt(len(w)))

    def proj(g):                                       # gradient with respect to conj(w) -> relative real variables
        if kind == "full":
            return 2.0 * rms * np.concatenate([g.real, g.imag])
        return 2.0 * np.concatenate([np.real(np.conj(g) * 1j * w)[1:], [np.real(np.vdot(g, w))]])
    _, g = _f_grad(a, w, target)
    gf = proj(g)
    cols = []
    if r is not None and p_max is not None and power(r, w) >= p_max * (1 - near):
        cols.append(proj(r @ w))
    if q is not None and sar_max is not None:
        s = quad(q, w)
        for k in np.flatnonzero(s >= sar_max * (1 - near))[:64]:
            cols.append(proj(q[k] @ w))
    if not cols:
        return float(np.linalg.norm(gf))
    c = np.stack(cols, axis=1)
    c = c / np.maximum(np.linalg.norm(c, axis=0), 1e-300)
    _, resid = scipy.optimize.nnls(c, -gf)
    return float(resid)


def random_starts(w0, n, seed=0):
    """``n`` starting vectors: w0 with random phases (uniform) and magnitudes (0.5 to 1.5 times) per channel, from a
    seeded generator. The problem is not convex; several starts find better local optima."""
    rng = np.random.default_rng(seed)
    w0 = np.asarray(w0, complex)
    return [w0 * np.exp(1j * rng.uniform(-np.pi, np.pi, len(w0))) * rng.uniform(0.5, 1.5, len(w0)) for _ in range(n)]


def tradeoff(a, target, w0, q, sar_limits, r=None, p_max=None, kind="full"):
    """Uniformity against local SAR: the problem solved for each limit of ``sar_limits`` (W/kg).

    The limits are visited from the tightest to the loosest and each solve also tries the previous solution as a
    start (it is feasible for a looser limit), so the error never rises as the limit is relaxed. Returns a list of
    dict(sar_max, f, cv, mean, peak_sar, power, w, fallback), in the order of ``sar_limits``."""
    order = np.argsort(sar_limits)
    out, prev = [None] * len(sar_limits), None
    for i in order:
        lim = float(sar_limits[i])
        best = solve(a, target, w0, r, p_max, q, lim, kind)
        if prev is not None:
            again = solve(a, target, prev, r, p_max, q, lim, kind)
            if again["f"] < best["f"]:
                best = again
        prev = best["w"]
        out[i] = dict(best, sar_max=lim)
    return out
