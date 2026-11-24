"""08 Noise: what bandwidth, matrix and averages cost, and the floor of a magnitude image.

Gradient-echo scans of the QA phantom; the noise is read from four circles in the empty corners of the image.
Noise enters at the receiver, per k-space sample. In the image its standard deviation sigma then scales with
matrix^1.5 (at a fixed field of view), bandwidth^0.5 and averages^-0.5. The magnitude of pure noise is Rayleigh
distributed: mean 1.2533 sigma, standard deviation 0.6551 sigma (Gudbjartsson & Patz 1995). Inside the object the
magnitude mean follows the Rician law and overestimates a weak signal; the complex mean does not.

    python examples/textbook/08_noise.py [--plot FILE.png]
"""
import _common as C
import numpy as np

BASE = dict(sequence="spgr", tr_ms=20, flip_deg=15, matrix=128, bandwidth_hz_per_px=300)
CORNERS = [dict(id=f"corner{i}", y_mm=92.0 * sy, x_mm=92.0 * sx, radius_mm=22.0) for i, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]
SWEEPS = dict(matrix=(64, 128, 256), bandwidth_hz_per_px=(150.0, 300.0, 600.0, 1200.0), averages=(1, 4, 16))
EXPECTED = dict(matrix=1.5, bandwidth_hz_per_px=0.5, averages=-0.5)


def background(readouts):
    """sigma (per real channel) from the second moment of the magnitude, and mean / sigma, SD / sigma of the magnitude."""
    n = np.array([r["n_pixels"] for r in readouts], float)
    mean, sd = C.column(readouts, "mean_magnitude")[:, 0], C.column(readouts, "sd_magnitude")[:, 0]
    sigma = np.sqrt(np.sum(n * (sd ** 2 * (n - 1) / n + mean ** 2)) / n.sum() / 2)          # E|noise|^2 = 2 sigma^2
    return float(sigma), float(np.sum(n * mean) / n.sum() / sigma), float(np.sqrt(np.sum(n * sd ** 2) / n.sum()) / sigma)


def rician_mean(a, sigma, order=80):
    """Mean magnitude of a + complex Gaussian noise, by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite.hermgauss(order)
    return float((np.outer(w, w) * np.hypot(a + np.sqrt(2) * sigma * x[:, None], np.sqrt(2) * sigma * x[None, :])).sum() / np.pi)


def main():
    args = C.parse_args(__doc__)
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    curves, rayleigh = {}, []
    for name, values in SWEEPS.items():
        stats = [background(C.scan(env, CORNERS, **dict(BASE, **{name: v}))["roi_readouts"]) for v in values]
        sigmas = np.array([s[0] for s in stats])
        rayleigh += [(s[1], s[2]) for s in stats]
        slope = float(np.polyfit(np.log(values), np.log(sigmas), 1)[0])
        curves[name] = (values, sigmas)
        C.check(f"exponent of sigma against {name}", slope, EXPECTED[name], 0.05)
    C.check("background magnitude, mean / sigma (Rayleigh 1.2533)", float(np.mean([r[0] for r in rayleigh])), 1.2533, 0.02)
    C.check("background magnitude, SD / sigma (Rayleigh 0.6551)", float(np.mean([r[1] for r in rayleigh])), 0.6551, 0.02)

    # a weak signal: 1 degree flip angle, small pixels, wide bandwidth
    vials = C.water_regions(world)
    weak = C.scan(env, vials + CORNERS, **dict(BASE, flip_deg=1.0, matrix=256, bandwidth_hz_per_px=1200))["roi_readouts"]
    sigma = background(weak[len(vials):])[0]
    ratio, floor = [], []
    for r in weak[:len(vials)]:
        a, m = r["complex_mean_magnitude"][0], r["mean_magnitude"][0]
        ratio.append(m / rician_mean(a, sigma))
        floor.append(m / a)
    print(f"weak signal, complex mean / sigma {min(r['complex_mean_magnitude'][0] for r in weak[:len(vials)]) / sigma:.2f} to "
          f"{max(r['complex_mean_magnitude'][0] for r in weak[:len(vials)]) / sigma:.2f}: magnitude mean / complex mean {min(floor):.2f} to {max(floor):.2f}")
    assert min(floor) > 1.2, "the magnitude mean must overestimate a signal near the noise level"
    C.check("largest |magnitude mean / Rician expectation - 1|", max(abs(v - 1) for v in ratio), 0.0, 0.06)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, (name, (values, sigmas)) in zip(axes, curves.items()):
            ax.loglog(values, sigmas, "o-")
            ax.loglog(values, sigmas[0] * (np.array(values) / values[0]) ** EXPECTED[name], "k:", lw=1)
            ax.set(xlabel=name, ylabel="image noise sigma", title=f"dotted: slope {EXPECTED[name]}")
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
