"""Estimators used on measured signals and on engine images alike, each written to follow the published analysis of
its data set.

ir_rdnls     Barral et al. 2010, reduced-dimension non-linear least squares for S(TI) = a + b exp(-TI / T1), the fit
             of the ISMRM 2020 T1 challenge: complex data (a, b complex), or magnitude data with polarity restoration
             (the two sign patterns round the minimum are tried and the better fit kept).
ir_three     the same three-parameter form fitted to signed ROI means by least squares; returns (a, b, T1, residuals).
vfa_linear   Fram 1987 / Deoni 2003: the line through S / sin(a) against S / tan(a).
decay        S0 exp(-TE / T2) by least squares on the echoes given.
"""
import numpy as np
import scipy.optimize

T1_GRID = np.concatenate([np.arange(1.0, 100.0, 0.25), np.arange(100.0, 5001.0, 1.0)])


def _rdnls(y, ti, grid):
    """y [..., n_ti] real or complex. For every candidate T1 the linear parameters are solved in closed form;
    returns (T1 of the smallest residual, that residual)."""
    x = np.exp(-ti[None, :] / grid[:, None])
    xc = x - x.mean(1, keepdims=True)
    xn = xc / np.linalg.norm(xc, axis=1, keepdims=True)
    yc = y - y.mean(-1, keepdims=True)
    proj = np.abs(yc @ xn.T.conj()) ** 2                           # energy explained by each candidate
    k = proj.argmax(-1)
    res = (np.abs(yc) ** 2).sum(-1) - np.take_along_axis(proj, k[..., None], -1)[..., 0]
    return grid[k], res


def ir_rdnls(y, ti_ms, magnitude=False, grid=T1_GRID):
    ti = np.asarray(ti_ms, float)
    y = np.asarray(y)
    if not magnitude:
        return _rdnls(y, ti, grid)[0]
    m = np.abs(y)
    kmin = m.argmin(-1)
    idx = np.arange(ti.size)
    best_t1, best_res = None, None
    for inclusive in (False, True):                                # flip the points before (or up to) the minimum
        flip = (idx[None, :] <= kmin[..., None]) if inclusive else (idx[None, :] < kmin[..., None])
        t1, res = _rdnls(np.where(flip, -m, m), ti, grid)
        if best_t1 is None:
            best_t1, best_res = t1, res
        else:
            better = res < best_res
            best_t1, best_res = np.where(better, t1, best_t1), np.where(better, res, best_res)
    return best_t1


def ir_three(ti_ms, s, starts=(100.0, 400.0, 1500.0)):
    """a + b exp(-TI / T1) fitted to signed signals: (a, b, T1, residuals)."""
    ti, s = np.asarray(ti_ms, float), np.asarray(s, float)
    best = None
    for t0 in starts:
        r = scipy.optimize.least_squares(lambda p: p[0] + p[1] * np.exp(-ti / p[2]) - s, [s[-1], -2 * s[-1], t0],
                                         bounds=([-np.inf, -np.inf, 1.0], [np.inf, np.inf, 20000.0]), x_scale=[abs(s[-1]) + 1e-30, abs(s[-1]) + 1e-30, 100.0])
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(best.x[1]), float(best.x[2]), best.fun


def restore_polarity(ti_ms, mag):
    """Magnitudes of an inversion recovery with the sign before the null restored: the sign pattern (points before,
    or up to, the smallest one flipped) whose three-parameter fit is better."""
    mag = np.asarray(mag, float)
    k, idx, best = int(mag.argmin()), np.arange(mag.size), None
    for flip in (idx < k, idx <= k):
        y = np.where(flip, -mag, mag)
        cost = float((ir_three(ti_ms, y)[3] ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, y)
    return best[1]


def vfa_linear(s, flip_rad, tr_ms):
    """s [..., n_flip] magnitudes; flip_rad the angles ASSUMED by the fit (nominal)."""
    a = np.broadcast_to(np.asarray(flip_rad, float), s.shape)
    yy, xx = s / np.sin(a), s / np.tan(a)
    xm, ym = xx.mean(-1, keepdims=True), yy.mean(-1, keepdims=True)
    slope = ((xx - xm) * (yy - ym)).sum(-1) / ((xx - xm) ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((slope > 0) & (slope < 1), -tr_ms / np.log(slope), np.nan)


def decay(te_ms, s, t2_start=50.0):
    """S0 exp(-TE / T2) fitted by least squares: (S0, T2, residuals)."""
    te, s = np.asarray(te_ms, float), np.asarray(s, float)
    r = scipy.optimize.least_squares(lambda p: p[0] * np.exp(-te / p[1]) - s, [s[0] * 1.2, t2_start], bounds=([0, 0.5], [np.inf, 5000.0]),
                                     x_scale=[s[0] + 1e-30, 50.0])
    return float(r.x[0]), float(r.x[1]), r.fun
