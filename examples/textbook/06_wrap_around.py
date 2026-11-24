"""06 Wrap-around: an object larger than the phase field of view folds over; the readout direction does not fold.

The QA phantom is about 190 mm across. Scans with the same 3.75 mm pixels:
  full     field of view 240 mm, nothing folds
  half     phase field of view 120 mm: the image is the full image plus its copy shifted by 120 mm along phase
  half+os  the same with 100 % phase oversampling: the fold is gone, the image is the cropped full image
  small    field of view 120 mm in both directions: folds along phase only (the readout is band-filtered)
  half, x  phase encoding along x: the fold follows the phase-encode axis
The fold law is checked on region read-outs: (region in the folded scan) = (region) + (region 120 mm away) in the
full scan, as complex numbers.

    python examples/textbook/06_wrap_around.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import world as W

BASE = dict(sequence="spgr", tr_ms=20, flip_deg=15, matrix=64)
PROBES = [dict(id="a", y_mm=-45.0, x_mm=0.0, radius_mm=8.0), dict(id="b", y_mm=45.0, x_mm=0.0, radius_mm=8.0),
          dict(id="c", y_mm=-20.0, x_mm=-45.0, radius_mm=8.0), dict(id="d", y_mm=20.0, x_mm=45.0, radius_mm=8.0)]
SHIFT = 120.0


def read(env, scan, regions):
    rows = C.call(env, "read_regions", scan=scan["scan_index"], regions=regions)["roi_readouts"]
    return np.array([r["complex_mean_magnitude"][0] * np.exp(1j * r["complex_mean_phase_rad"][0]) if r["n_pixels"] else 0.0
                     for r in rows])


def partners(key):
    """The same circles moved by one (half) field of view towards the other side of the isocentre."""
    return [dict(p, **{key: p[key] - np.sign(p[key]) * SHIFT}) for p in PROBES]


def main():
    args = C.parse_args(__doc__)
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    scans = dict(full=C.scan(env, PROBES, **BASE), half=C.scan(env, PROBES, **dict(BASE, phase_fov=0.5)),
                 half_os=C.scan(env, PROBES, **dict(BASE, phase_fov=0.5, phase_oversampling=1.0)),
                 small=C.scan(env, PROBES, **dict(BASE, fov_mm=120, matrix=32)),
                 full_x=C.scan(env, PROBES, **dict(BASE, pe_dir="x")), half_x=C.scan(env, PROBES, **dict(BASE, phase_fov=0.5, pe_dir="x")))
    full, full_x = read(env, scans["full"], PROBES), read(env, scans["full_x"], PROBES)
    fold_y = full + read(env, scans["full"], partners("y_mm"))
    fold_x = full_x + np.where([p["x_mm"] != 0 for p in PROBES], read(env, scans["full_x"], partners("x_mm")), 0.0)
    unit = np.abs(full).max()
    box = C.call(env, "view", scan=scans["full"]["scan_index"])["bounding_box"]
    truth = W.extent_mm(world)
    print(f"full field of view: object seen at y {box['y_mm'][0]:.1f} .. {box['y_mm'][1]:.1f} mm (true {truth['y'][0]:.1f} .. {truth['y'][1]:.1f}), "
          f"touches the border: {box['touches_edge']}")
    assert not box["touches_edge"] and max(abs(box["y_mm"][i] - truth["y"][i]) for i in (0, 1)) <= 3.75
    half = read(env, scans["half"], PROBES)
    print(f"phase field of view 120 mm: touches the border: {C.call(env, 'view', scan=scans['half']['scan_index'])['bounding_box']['touches_edge']}; "
          f"regions a, b change by {np.abs(half - full)[:2].min() / unit:.2f} to {np.abs(half - full)[:2].max() / unit:.2f} of the largest region signal")
    assert np.abs(half - full)[:2].min() > 0.3 * unit, "the folded copy must be visible in regions a and b"
    C.check("half: largest |folded - (region + region 120 mm away)| / largest signal", np.abs(half - fold_y).max() / unit, 0.0, 0.03)
    C.check("half + phase oversampling: largest |image - full image| / largest signal",
            np.abs(read(env, scans["half_os"], PROBES) - full).max() / unit, 0.0, 0.03)
    small = read(env, scans["small"], PROBES)
    C.check("small: largest |image - fold along phase only| / largest signal", np.abs(small - fold_y).max() / unit, 0.0, 0.03)
    other = full + read(env, scans["full"], partners("x_mm"))[[2, 3, 2, 3]] * np.array([0, 0, 1, 1])
    print(f"small: a fold along the readout would have changed regions c, d by {np.abs(other - full)[2:].min() / unit:.2f} to "
          f"{np.abs(other - full)[2:].max() / unit:.2f}; measured change {np.abs(small - full)[2:].max() / unit:.3f}")
    assert np.abs(other - full)[2:].min() > 0.3 * unit and np.abs(small - full)[2:].max() < 0.03 * unit
    C.check("half, phase encoding along x: largest |folded - fold along x| / largest signal",
            np.abs(read(env, scans["half_x"], PROBES) - fold_x).max() / unit, 0.0, 0.03)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mrgym.scanner import kspace
        engine = kspace.Engine(world)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
        for ax, (name, s) in zip(axes, [(k, scans[k]) for k in ("full", "half", "half_os", "small", "half_x")]):
            fy, fx = s["fov_yx_mm"]
            ax.imshow(np.abs(engine.acquire(s["realised"])[0]), cmap="gray", extent=(-fx / 2, fx / 2, fy / 2, -fy / 2))
            ax.set(title=name, xlim=(-120, 120), ylim=(120, -120), xlabel="x (mm)")
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
