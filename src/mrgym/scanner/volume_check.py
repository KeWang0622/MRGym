"""Checks of the volume worlds (``volume.py``) and the page that reports them (``docs/physics/volume.md``).

    python -m mrgym.scanner.volume_check --write     measure, store docs/physics/volume.json, write the page
    python -m mrgym.scanner.volume_check --doc       write the page from the stored numbers
    python -m mrgym.scanner.volume_check             print the measurement

Each function below is one check: ``tests/test_volume_world.py`` asserts on what it returns, ``measure`` collects
the same numbers for the page. Nothing on the page is typed: it is ``doc_page`` of the stored measurement
(``tests/test_volume_world.py::test_the_page_is_the_stored_measurement``). Times and memory are those of the machine
and the moment of the measurement (one thread); every other number is a function of the code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import tracemalloc

import numpy as np

from . import anatomy, volume
from . import world as W

SEEDS = tuple("%032x" % (k * 1000003 + 17) for k in range(4))          # fixed 128-bit seeds of the checks
N_RANGE = 40                                                            # worlds drawn for the table of what varies
EDGE = anatomy.EDGE_MM
INSIDE_MM = 4.0                                                         # 'inside the object': this far from any air
OBLIQUE = dict(name="axial", pos_mm=5.0, tilt_deg=(12.0, -8.0))        # the plane of the plane-against-points check
PLANES = dict(axial=dict(name="axial"), sagittal=dict(name="sagittal"), coronal=dict(name="coronal"),
              oblique=dict(name="axial", pos_mm=10.0, tilt_deg=(20.0, 15.0)))
LIMITS = dict(plane_points=1e-4, weight_sum=1e-6, common_line=1e-4, lattice_volume=5e-3, flat_2d=1e-6, build_s=1.2, plane_ms=100.0)


def make(kind, seed=SEEDS[0], **spec):
    """A volume world of the checks (the head with its lesions)."""
    return W.make_world(seed, dict(dict(kind=kind, lesion="ms" if kind == "head3d" else None), **spec))


# ---------------------------------------------------------------------------------------------- the checks
def plane_against_points(vol, frame=None, n=256, extent_mm=256.0) -> dict:
    """``eval_plane`` (rectangles, matrix products, the precision of the kind) against ``eval_points`` (the recipe,
    float64, no shortcut) at the same points: largest and rms difference of a class weight."""
    frame = frame or volume.Frame.plane(**OBLIQUE)
    c = volume.grid_axis(n, extent_mm)
    d = vol.eval_plane(frame, n, extent_mm, solid=True) - vol.eval_points(*frame.points(c, c), solid=True)
    return dict(max=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d * d))))


def weight_sums(vol, frames=None, n=320) -> dict:
    """Over several planes: the largest sum of all matter classes and the smallest weight. And on the 2 mm lattice:
    how far the sum is from 1 inside the object, i.e. at the voxels that are at least ``INSIDE_MM`` from anything
    that is not matter (found on the same lattice drawn with sharp edges, eroded in 3D)."""
    from scipy import ndimage
    frames = frames or [volume.Frame.plane(**p) for p in PLANES.values()] + [volume.Frame.plane("axial", z) for z in (-70.0, -30.0, 40.0)]
    top, low = 0.0, 0.0
    for f in frames:
        w = vol.eval_plane(f, n, solid=True)
        top, low = max(top, float(w.sum(0, dtype=np.float64).max())), min(low, float(w.min()))
    step = 2.0
    sharp = vol.lattice(step, labels=True, edge_mm=1e-3).weights > 0
    k = int(np.ceil(INSIDE_MM / step + np.sqrt(3.0)))                    # a sharp boundary lies within a voxel diagonal of where the lattice sees it
    zz, yy, xx = np.mgrid[-k:k + 1, -k:k + 1, -k:k + 1]
    core = ndimage.binary_erosion(sharp, structure=zz * zz + yy * yy + xx * xx <= k * k, border_value=1)
    total = vol.lattice(step).weights.sum(0, dtype=np.float64)
    top = max(top, float(total.max()))
    return dict(max_sum=top, min_weight=low, inside_off_one=float(np.abs(total[core] - 1.0).max()), inside_voxels=int(core.sum()))


def common_line(vol, origin=(4.0, -6.0, 12.0), n=320) -> float:
    """Two planes through one line (the row axis through ``origin``; the second plane tilted 35 degrees about it)
    hold the same points on that line: largest difference of a class weight there. Then the same for an axial and a
    sagittal plane, whose grids share the points of their line of intersection."""
    a = volume.Frame.plane("axial", origin=origin)
    b = volume.Frame.plane("axial", origin=origin, tilt_deg=(35.0, 0.0))
    d1 = np.abs(vol.eval_plane(a, n, solid=True)[:, :, n // 2] - vol.eval_plane(b, n, solid=True)[:, :, n // 2]).max()
    step = vol.grid[1] / n
    z0, x0 = 20 * step, -15 * step                                      # grid positions of both planes
    ax = vol.eval_plane(volume.Frame.plane("axial", z0), n, solid=True)[:, :, n // 2 - 15]            # x = x0, all y
    sg = vol.eval_plane(volume.Frame.plane("sagittal", x0), n, solid=True)[:, n // 2 - 20, :]          # row -z: z = z0, all y
    return float(max(d1, np.abs(ax - sg).max()))


def ball_volume(radius, edge=EDGE):
    """Volume of a ball with the soft edge of the worlds: (4/3) pi R^3 (1 + (pi^2 / 4) (e / R)^2), exact for
    0.5 (1 + tanh(d / e)) up to terms exp(-2R / e)."""
    return 4.0 / 3.0 * np.pi * radius ** 3 * (1.0 + np.pi ** 2 / 4.0 * (edge / radius) ** 2)


def disc_area(radius, edge=EDGE):
    """Area of a disc with the same soft edge: pi R^2 (1 + (pi^2 / 12) (e / R)^2)."""
    return np.pi * radius ** 2 * (1.0 + np.pi ** 2 / 12.0 * (edge / radius) ** 2)


def lattice_volumes(spacing=2.0) -> list:
    """Volume of a ball and of a cylinder summed on the lattice against the closed form, for the two ways a lattice
    is sampled: the value at the voxel centre (the default) and the mean of 2 x 2 x 2 samples per voxel.

    Hand-made shapes of radius 24 mm, the cylinder along z with flat ends, each at five offsets of its centre
    within a voxel (the worst is kept): against the sharp-edged form and the soft-edged one. Then the ball and one
    vial of ``qa3d`` (smaller, so the soft edge counts): against the soft-edged form, the object that is drawn."""
    r, hl = 24.0, 40.3
    box = dict(z=(-70.0, 70.0), y=(-40.0, 40.0), x=(-40.0, 40.0))
    offsets = [(0.31 * k * spacing / 4, -0.7 + 0.2 * k, k * spacing / 4) for k in range(5)]
    rows = []

    def row(shape, sampling, got, sharp, soft):
        worst = max(got, key=lambda v: abs(v / soft - 1))
        rows.append(dict(shape=shape, sampling=sampling, spacing_mm=spacing, lattice_mm3=worst, sharp_mm3=sharp, soft_mm3=soft,
                         rel_sharp=worst / sharp - 1, rel_soft=worst / soft - 1))

    w = make("qa3d")
    vol, lay = w["volume"], w["volume"].layout
    rs, vial = lay["sphere"]["radius"], lay["vials"][0]
    area, length = np.pi * vial["a"] * vial["b"], 2 * lay["vial_half_length"]
    for k, sampling in ((1, "voxel centre"), (2, "mean of 2 x 2 x 2")):
        ball, cyl = [], []
        for c in offsets:
            b = volume.shapes_volume([("wm", lambda y, x, z, c=c: volume.sd_sphere(y, x, z, c, r))])
            t = volume.shapes_volume([("wm", lambda y, x, z, c=c: volume.sd_cylinder(y, x, z, c, (0.0, 0.0, 1.0), r, hl))])
            ball.append(float(b.lattice(spacing, box=box, supersample=k).weights.sum(dtype=np.float64)) * spacing ** 3)
            cyl.append(float(t.lattice(spacing, box=box, supersample=k).weights.sum(dtype=np.float64)) * spacing ** 3)
        row("ball, radius 24 mm", sampling, ball, 4.0 / 3.0 * np.pi * r ** 3, ball_volume(r))
        row(f"cylinder along z, radius 24 mm, length {2 * hl:g} mm", sampling, cyl, np.pi * r * r * 2 * hl, disc_area(r) * 2 * hl)
        lat = vol.lattice(spacing, supersample=k)
        row(f"the ball of qa3d, radius {rs:.2f} mm", sampling, [float(lat.weights[vol.matter.index("sphere")].sum(dtype=np.float64)) * spacing ** 3],
            4.0 / 3.0 * np.pi * rs ** 3, ball_volume(rs))
        row(f"vial 0 of qa3d (along z), semi-axes {vial['a']:.2f} x {vial['b']:.2f} mm, length {length:.2f} mm", sampling,
            [float(lat.weights[vol.matter.index("vial0")].sum(dtype=np.float64)) * spacing ** 3], area * length,
            area * (1.0 + np.pi ** 2 / 12.0 * EDGE ** 2 / (vial["a"] * vial["b"])) * length)
    return rows


def flat_plane_against_2d(seed, qa=None, rot_deg=0.0) -> dict:
    """The reference plane of ``qa3d`` (axial, through the isocentre: inside its vial section) against the 2D
    ``qa`` WORLD of the same seed and spec, built by ``anatomy.qa``: the largest difference of a class weight,
    whether the arrays are equal bit for bit, whether the 3D classes are absent from the plane, and whether tissue
    values and regions are the 2D ones."""
    flat, vol = W.make_world(seed, dict(kind="qa", qa=qa)), W.make_world(seed, dict(kind="qa3d", qa=qa))
    if rot_deg:
        flat, vol = W.turned(flat, rot_deg), W.turned(vol, rot_deg)
    k = len(flat["names"])
    two = [(r["id"], r["cls"], r["y_mm"], r["x_mm"], r["radius_mm"]) for r in flat["regions"]]
    three = [(r["id"], r["cls"], r["y_mm"], r["x_mm"], r["radius_mm"]) for r in vol["regions"][:len(two)]]
    return dict(max=float(np.abs(flat["weights"] - vol["weights"][:k]).max()), bitwise=bool(np.array_equal(flat["weights"], vol["weights"][:k])),
                rest_absent=not vol["weights"][k:].any(), names=flat["names"] == vol["names"][:k],
                tissues=all(flat["tissues"][n] == vol["tissues"][n] for n in flat["names"]), regions=two == three)


def timings(kind, seeds=SEEDS[:3], repeats=5) -> dict:
    """Build time of a world (make_world: recipe, regions, reference plane) and the time of one plane on the master
    grid; the best of ``repeats`` (a shared machine adds, it never subtracts)."""
    build, recipe = [], []
    for seed in seeds:
        t = time.perf_counter()
        w = make(kind, seed)
        build.append(time.perf_counter() - t)
        t = time.perf_counter()
        volume.make_volume(seed, w["source"]["spec"])
        recipe.append(time.perf_counter() - t)
    vol, plane = w["volume"], {}
    for name, p in PLANES.items():
        f, ts = volume.Frame.plane(**p), []
        for _ in range(repeats):
            t = time.perf_counter()
            vol.eval_plane(f)
            ts.append(time.perf_counter() - t)
        plane[name] = 1e3 * min(ts)
    return dict(build_s=[round(v, 3) for v in build], recipe_and_regions_s=[round(v, 3) for v in recipe], grid=list(vol.grid),
                plane_ms={k: round(v, 1) for k, v in plane.items()})


def memory(kind, seed=SEEDS[0]) -> dict:
    """Peak of Python-side allocations (tracemalloc, which counts NumPy arrays) while a world is built, while one
    plane is evaluated, and while the 2 mm lattice is built; and what the world and the lattice hold afterwards."""
    out = {}
    tracemalloc.start()
    w = make(kind, seed)
    out["build_peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6
    out["world_held_mb"] = tracemalloc.get_traced_memory()[0] / 1e6
    tracemalloc.reset_peak()
    w["volume"].eval_plane(volume.Frame.plane("sagittal"))
    out["plane_peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6 - out["world_held_mb"]
    tracemalloc.reset_peak()
    t = time.perf_counter()
    lat = w["volume"].lattice(2.0)
    out["lattice_2mm_s"] = time.perf_counter() - t
    out["lattice_2mm_peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6 - out["world_held_mb"]
    tracemalloc.stop()
    out["lattice_2mm_shape"], out["lattice_2mm_held_mb"] = list(lat.weights.shape), lat.weights.nbytes / 1e6
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


def digest(world) -> str:
    """sha256 of everything a volume world is: its reference plane, two other planes, coarse lattice labels, and
    the JSON of layout, regions, tissues, system state and truth. Equal digests = the same world, bit for bit."""
    vol = world["volume"]
    h = hashlib.sha256(np.ascontiguousarray(world["weights"]).tobytes())
    for p in ("sagittal", "oblique"):
        h.update(vol.eval_plane(volume.Frame.plane(**PLANES[p]), 160, solid=True).tobytes())
    h.update(vol.lattice(8.0, labels=True).weights.tobytes())
    plain = dict(layout=vol.layout, regions=vol.regions, tissues=vol.tissues, system=world["system"], truth=vol.truth(), chi=vol.susceptibility_ppm,
                 names=vol.matter, console=world["console"])
    h.update(json.dumps(plain, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else list(o)).encode())
    return h.hexdigest()


def hidden_ranges(kind, n=N_RANGE) -> dict:
    """What a seed draws: the smallest and the largest value of each hidden quantity over ``n`` worlds (``layout_only``
    builds of seeds ``range:<k>`` hashed to 128 bits)."""
    rows = {}

    def see(name, value):
        lo, hi = rows.get(name, (value, value))
        rows[name] = (min(lo, value), max(hi, value))

    for k in range(n):
        seed = hashlib.sha256(f"range:{k}".encode()).hexdigest()[:32]
        w = W.make_world(seed, dict(kind=kind, lesion="ms" if kind == "head3d" else None), layout_only=True)
        lay = w["layout"]
        see("scored regions", len(w["regions"]))
        if kind == "head3d":
            for key, label in (("ay", "half-length of the head, front to back (mm)"), ("ax", "half-width of the head (mm)"), ("az", "height of the head above its centre (mm)"),
                               ("zc", "centre of the head above the isocentre (mm)"), ("neck", "width of the neck / width of the head"),
                               ("fat", "scalp fat, mean thickness (mm)"), ("bone", "skull, mean thickness (mm)"), ("b_csf", "CSF over the brain (mm)"),
                               ("t_gm", "cortex thickness (mm)"), ("atrophy", "atrophy (0 to 1: wider sulci and ventricles)")):
                see(label, lay[key])
            see("lesions", len(lay["lesions"]))
            see("susceptibility of the lesions, size (ppm)", abs(lay["lesion_chi_ppm"]))
            see("susceptibility of the lesions, sign", float(np.sign(lay["lesion_chi_ppm"])))
            for v in lay["vessels"]:
                see("vessel radius (mm)", v["radius"])
                see("vessel speed, size (mm/s)", abs(v["speed_mm_s"]))
                see("vessel speed, sign along its direction", float(np.sign(v["speed_mm_s"])))
                see("vessel direction, angle from the z axis (degrees)", float(np.degrees(np.arccos(abs(v["direction"][2])))))
            for s in lay["air"]:
                if "c_mm" in s:
                    see(f"{s['name'].rstrip('_LR').replace('_', ' ')}: centre along z (mm)", s["c_mm"][2])
            see("thalamus (left): centre along z (mm)", next(s["c_mm"][2] for s in lay["deep_nuclei"] if s["name"] == "thalamus_L"))
        else:
            b = lay["body"]
            for label, value in (("body: semi-axes (mm)", b["a"]), ("body: semi-axes (mm)", b["b"]), ("body: half-length (mm)", b["half_length"]),
                                 ("body: centre along z (mm)", b["z"]), ("vials", len(lay["vials"])), ("vial section: half-length (mm)", lay["vial_half_length"]),
                                 ("side of the ball and the wedges along z (+1 or -1); the ramp block is on the other", lay["side_of_sphere_and_wedges"]),
                                 ("ball: radius (mm)", lay["sphere"]["radius"]), ("ball: centre along z (mm)", lay["sphere"]["c"][2]),
                                 ("ball: distance of its centre from the axis (mm)", float(np.hypot(lay["sphere"]["c"][0] - b["y"], lay["sphere"]["c"][1] - b["x"]))),
                                 ("wedges: angle of the tilted faces to the axial plane (degrees)", lay["wedges"]["angle_deg"]),
                                 ("wedges: orientation in the plane (degrees)", lay["wedges"]["rot_deg"]),
                                 ("ramps: rise per mm along the ramp", lay["ramp_block"]["slope"]), ("ramp block: orientation in the plane (degrees)", lay["ramp_block"]["rot_deg"]),
                                 ("ramp block: centre along z (mm)", lay["ramp_block"]["c"][2])):
                see(label, value)
    return [[k, round(float(a), 3), round(float(b), 3)] for k, (a, b) in rows.items()]


# ---------------------------------------------------------------------------------------------- the measurement and its page
def measure() -> dict:
    out = dict(engine=__import__("mrgym").ENGINE_VERSION, numpy=np.__version__, machine=platform.machine(), python=platform.python_version(),
               limits=LIMITS, seeds=list(SEEDS), kinds={})
    for kind in volume.KINDS:
        w = make(kind)
        vol = w["volume"]
        pp = [plane_against_points(make(kind, s)["volume"]) for s in SEEDS]
        row = dict(names=list(vol.names), solid=list(vol.solid), organ=list(vol.organ), regions=len(vol.regions), precision=str(vol._dtype),
                   plane_points=dict(max=max(p["max"] for p in pp), rms=max(p["rms"] for p in pp), grid=256, seeds=len(SEEDS)),
                   sums=weight_sums(vol), common_line=common_line(vol), timings=timings(kind), memory=memory(kind),
                   chi_ppm={k: v for k, v in vol.susceptibility_ppm.items() if v}, ranges=hidden_ranges(kind), layout_keys=sorted(vol.layout))
        out["kinds"][kind] = row
    out["lattice_volumes"] = lattice_volumes()
    flat = [flat_plane_against_2d(s, qa, rot) for s in SEEDS for qa in (None, dict(n_oil=2), dict(body_aspect=(1.2, 1.5))) for rot in (0.0, 30.0)]
    out["flat_2d"] = dict(cases=len(flat), max=max(f["max"] for f in flat), bitwise=all(f["bitwise"] for f in flat),
                          all_else=all(f[k] for f in flat for k in ("rest_absent", "names", "tissues", "regions")))
    try:
        import resource
        out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)
    except ImportError:
        out["peak_rss_mb"] = None
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else list(o)))


def _g(v, digits=3):
    return f"{v:.{digits}g}"


def doc_page(m: dict) -> str:
    """``docs/physics/volume.md`` from a stored measurement."""
    head, qa, lim = m["kinds"]["head3d"], m["kinds"]["qa3d"], m["limits"]
    lines = [
        "# Volume worlds: the third dimension of the hidden object", "",
        f"Generated by `python -m mrgym.scanner.volume_check --write` (engine `{m['engine']}`, NumPy {m['numpy']}, {m['machine']}, one thread) "
        "from `docs/physics/volume.json`. Do not edit: regenerate. Times and memory are those of the machine of that run.", "",
        "A 2D world is one axial section. The world kinds `head3d` and `qa3d` are objects with structure along the third axis "
        "(`src/mrgym/scanner/volume.py`). This page says what they are and how they are evaluated. **No scan is formed of a volume world "
        "yet**: a volume world has no fields, and the engine images 2D worlds only. Fields in 3D and the slice engine are the next packages.", "",
        "## 1. Coordinates", "",
        "Patient coordinates in mm from the isocentre. A vector is written (y, x, z), the order of the engine's arrays: "
        "+y is posterior, +x is the patient's left, +z is superior (towards the top of the head). B0 is along +z. "
        "(x, y, z) is the right-handed DICOM patient system (left, posterior, superior).", "",
        "```text",
        "            +z superior (head)                   axial image (row +y, col +x)",
        "             |                                         anterior",
        "             |                                   +----------------+",
        "             |                                   |                |",
        "             o--------- +x  patient's left     R |    (y, x)      | L     row index grows downwards (+y)",
        "            /                                    |                |       column index grows to the right (+x)",
        "           /                                     +----------------+",
        "         +y posterior                                 posterior",
        "```", "",
        "A plane is a `Frame`: an origin and three orthonormal vectors `row`, `col`, `normal`. Grid point (i, j) of an n x n grid of extent E lies at "
        "origin + (i - n // 2) (E / n) row + (j - n // 2) (E / n) col. That is the engine's in-plane rule, so the axial frame through the isocentre "
        "is the grid of every 2D world.", "",
        "| Standard plane | row (down on the image) | col (to the right) | normal |", "|---|---|---|---|",
        "| axial | +y: anterior at the top | +x: the patient's left on the right | +z |",
        "| coronal | -z: the top of the head at the top | +x | +y |",
        "| sagittal | -z | +y: anterior on the left | +x |", "",
        "The normal of a standard plane is the positive patient axis, so a position along it is the patient coordinate. `rot_deg` turns a frame in "
        "its plane as `world.turned` does; `tilt_deg` tilts it about its row axis, then about its column axis. A lattice is an array [z, y, x]; "
        "its origin and spacing are given per array axis.", "",
        "## 2. What the two recipes hold", "",
        "A world is a recipe: shapes with a signed distance (ellipsoids, tubes, planes, layers of a depth below a surface), drawn from the seed. "
        "Nothing is stored as voxels. Boundaries are soft with the half-width of the 2D worlds "
        f"({EDGE:g} mm), so a point near a boundary holds two classes whose weights sum to 1.", "",
        f"**`head3d`** ({len(head['names'])} classes with the lesions: {', '.join(head['names'])}). An egg-shaped head behind a smooth warp, narrowing "
        "below its centre into a neck that goes on downwards out of the imaged cube. From outside in: skin, scalp fat with a varying thickness, "
        "then either the cranial cavity or its shell. The shell is skull (two tables with fatty marrow between them) within the bone thickness of the "
        "cavity and muscle elsewhere (face and neck), with two eyes (drawn with the CSF values) and air: a frontal and a sphenoid sinus, two mastoid "
        "regions and an airway. The cavity holds CSF and the brain: cortex and white matter, sulci as the zero sheets of a smooth random field "
        "limited in depth, an interhemispheric fissure interrupted by a callosal region, three ellipsoids per lateral ventricle and a third "
        "ventricle, three deep nuclei per side. Options: lesions (`lesion=\"ms\"`: seeded at points the recipe itself reports as white matter, with a "
        "hidden sign and size of their susceptibility), straight vessels from the neck into the brain, each with a hidden direction and a hidden signed "
        "speed (`volume=dict(vessels=n)`), placed inserts (`volume=dict(inserts=[...])`), and a pose (`volume=dict(pose_deg=...)`).", "",
        f"**`qa3d`** (classes: {', '.join(qa['names'])}; without signal: {', '.join(qa['solid'])}). A cylinder of gel with flat ends. "
        "About the isocentre lies the vial section: tubes along the axis, laid out exactly as the 2D `qa` phantom of the same seed and spec "
        "(section 5). Beyond it on one side: a ball of its own material and a pair of wedges whose tilted faces cross (in an axial slice at a "
        "height d above their centre the two bars differ in length by 2 d / tan(angle)). On the other side: a block of plastic with two thin "
        "crossed ramps of fluid (in an axial slice of thickness T a ramp shows as a bar of length T / slope, and the two bars move apart with the "
        "slice position). Which side is which is drawn. Options: the 2D `qa` spec (oil vials, an elongated body, placed discs), "
        "`volume=dict(features=False)` for the vial section alone, placed inserts, a pose.", "",
        "### What a seed draws", "",
        f"Smallest and largest value over {N_RANGE} seeds (`volume_check.hidden_ranges`). Every number below is in `world[\"layout\"]` or "
        "`world[\"regions\"]` of the verifier's side; none is told to the agent.", "",
        "| `head3d`: hidden quantity | smallest | largest |", "|---|---|---|"]
    lines += [f"| {k} | {_g(a, 4)} | {_g(b, 4)} |" for k, a, b in head["ranges"]]
    lines += ["", "| `qa3d`: hidden quantity | smallest | largest |", "|---|---|---|"]
    lines += [f"| {k} | {_g(a, 4)} | {_g(b, 4)} |" for k, a, b in qa["ranges"]]
    lines += [
        "", "Also drawn: every tissue value (the head: the tissue table with the jitter of the 2D brain; the test object: the ranges of the 2D "
        "`qa` phantom), every semi-axis and rotation of the ventricles, nuclei, cavities and lesions, the in-plane positions of all inserts, "
        "the folding field of the sulci, and the ids of the scored regions (a random order: an id says nothing about what the region is).", "",
        "### Regions and truth", "",
        "`world[\"regions\"]` lists 3D regions: balls, and cylinders along an axis (`shape`, centre `y_mm, x_mm, z_mm`, `radius_mm`, for a cylinder "
        f"`axis` and `half_length_mm`). A region is listed only if its class fills at least {volume.PURITY_MIN:g} of it, measured on sample "
        "points of the recipe (`purity`). `world.truth(world)` returns the tissue values per region as for a 2D world; `world[\"volume\"].truth()` "
        "adds the susceptibility. `world[\"layout\"]` holds the geometry: positions are patient coordinates (the warp of the head is inverted "
        f"for them), sizes are those of the recipe. For this seed `head3d` lists {head['regions']} regions and `qa3d` {qa['regions']}.", "",
        "### Susceptibility", "",
        "Volume susceptibility relative to soft tissue, ppm (SI), per class (`volume.susceptibility_ppm`); classes not listed are 0. "
        "Air (9.4) is the engine's own constant (`world.DCHI_AIR`). **Every other value is declared**: chosen by the authors of this package, of the "
        "size the literature reports, with no source opened for this module.", "",
        "| Kind | Values |", "|---|---|",
        f"| `head3d` | {', '.join(f'{k} {v:+.2f}' for k, v in head['chi_ppm'].items())} (the lesion value is this seed's: its sign and size are drawn) |",
        f"| `qa3d` | {', '.join(f'{k} {v:+.2f}' for k, v in qa['chi_ppm'].items())}; gels and plastic 0 |", "",
        "## 3. What is schematic", "",
        "- This is drawn anatomy, not a segmented subject. There is no cerebellum, no brainstem, no cord, no falx or tentorium; the skull base is a "
        "shorter half of the same ellipsoid; the face and the neck are one muscle class; the neck is a cylinder that never widens into shoulders.",
        "- Vessels are straight tubes. They carry a direction and a signed speed as numbers; nothing flows until a slice engine uses them.",
        "- Sulci are sheets of a random field, not gyri with a name; the cortex has one thickness per head.",
        "- The eyes are balls of the CSF class. Air cavities are ellipsoids.",
        "- The test object is not a product: the vial section follows this package's 2D `qa` phantom, the ramps and wedges follow the idea of "
        "the ACR phantom (ramps of 1 in 10, crossed wedges) with drawn angles. The ACR dimensions belong to a later world (`acr3d`).",
        "- A lesion is one class with one susceptibility; a vessel has the susceptibility declared for venous blood whichever way it runs.", "",
        "## 4. How a plane is evaluated", "",
        "`eval_points(y, x, z)` evaluates the recipe directly, in float64: every shape at every point. It is the reference and the verifier's "
        "truth. `eval_plane(frame, n, extent_mm)` evaluates the same recipe on the engine's grid in a plane with two shortcuts that change no "
        "value beyond rounding:", "",
        "- a small shape is evaluated only inside the rectangle that its bounding box (plus the reach of the warp and of the soft edge) can touch "
        "on the plane;",
        "- a smooth random field is a sum of plane cosines, and on a plane cos(a + b v + c u) = Re(e^{ia} e^{ibv} e^{icu}): one small matrix "
        "product instead of a cosine per mode and point. Nothing is interpolated.", "",
        f"The head is evaluated in {head['precision']} on a plane (in strips of rows that stay in the cache), the test object in {qa['precision']}. "
        "A plane at any position, angle and thickness offset is therefore the object itself at those points; partial volume across a slice is "
        "the slice engine's sum over several such planes. `lattice(spacing_mm)` evaluates axial planes on a coarse lattice (values at the voxel "
        "centres; `supersample=k` averages k^3 samples per voxel), for the field of the object and for 3D acquisitions.", "",
        "## 5. Measured", "",
        f"Seeds `{m['seeds'][0]}` and the next {len(m['seeds']) - 1}; the limits are the acceptance limits of this package and are asserted in "
        "`tests/test_volume_world.py`.", "",
        "| Check | `head3d` | `qa3d` | Limit |", "|---|---|---|---|",
        f"| plane against points, largest difference of a class weight (grid {head['plane_points']['grid']}, a double-oblique plane, {head['plane_points']['seeds']} seeds) "
        f"| {_g(head['plane_points']['max'])} (rms {_g(head['plane_points']['rms'])}) | {_g(qa['plane_points']['max'])} (rms {_g(qa['plane_points']['rms'])}) | {lim['plane_points']:g} |",
        f"| largest sum of all classes at a point, minus 1 | {_g(head['sums']['max_sum'] - 1)} | {_g(qa['sums']['max_sum'] - 1)} | {lim['weight_sum']:g} |",
        f"| sum of all classes minus 1 at the lattice points at least {INSIDE_MM:g} mm inside the object | {_g(head['sums']['inside_off_one'])} | {_g(qa['sums']['inside_off_one'])} | {lim['weight_sum']:g} |",
        f"| two planes on their common line, largest difference | {_g(head['common_line'])} | {_g(qa['common_line'])} | {lim['common_line']:g} |",
        f"| build of a world (`make_world`), three seeds | {', '.join(f'{v:.2f}' for v in head['timings']['build_s'])} s | {', '.join(f'{v:.2f}' for v in qa['timings']['build_s'])} s | {lim['build_s']:g} s |",
        f"| one plane on the master grid ({head['timings']['grid'][0]} x {head['timings']['grid'][0]}, {head['timings']['grid'][1] / head['timings']['grid'][0]:g} mm): "
        "axial, sagittal, coronal, double-oblique | " + " / ".join(f"{head['timings']['plane_ms'][k]:.0f}" for k in PLANES) + " ms | "
        + " / ".join(f"{qa['timings']['plane_ms'][k]:.0f}" for k in PLANES) + f" ms | {lim['plane_ms']:g} ms |",
        f"| memory: peak while building / held by the world (tracemalloc) | {head['memory']['build_peak_mb']:.0f} / {head['memory']['world_held_mb']:.0f} MB "
        f"| {qa['memory']['build_peak_mb']:.0f} / {qa['memory']['world_held_mb']:.0f} MB | |",
        f"| memory: peak while one plane is evaluated | {head['memory']['plane_peak_mb']:.0f} MB | {qa['memory']['plane_peak_mb']:.0f} MB | |",
        "| lattice at 2 mm: shape [classes, z, y, x], time, bytes held, peak | "
        + " | ".join(f"{' x '.join(str(v) for v in k['memory']['lattice_2mm_shape'])}, {k['memory']['lattice_2mm_s']:.2f} s, {k['memory']['lattice_2mm_held_mb']:.0f} MB, "
                     f"{k['memory']['lattice_2mm_peak_mb']:.0f} MB" for k in (head, qa)) + " | |",
        "", f"Peak resident memory of the whole measurement (all of the above in one process): {m['peak_rss_mb']} MB.", "",
        f"Lattice volume against the closed form ({m['lattice_volumes'][0]['spacing_mm']:g} mm lattice). A soft edge of half-width e adds (pi^2 / 4) (e / R)^2 to "
        "the volume of a ball of radius R and (pi^2 / 12) (e / R)^2 to the area of a disc; the last column compares with that form, which is the "
        "object that is drawn. The hand-made shapes were placed at five offsets within a voxel; the worst is shown.", "",
        "| Shape | lattice sampling | lattice (mm^3) | against the sharp-edged form | against the soft-edged form |", "|---|---|---|---|---|"]
    lines += [f"| {r['shape']} | {r['sampling']} | {r['lattice_mm3']:.0f} | {100 * r['rel_sharp']:+.2f} % | {100 * r['rel_soft']:+.2f} % |" for r in m["lattice_volumes"]]
    point = [r for r in m["lattice_volumes"] if r["sampling"] == "voxel centre"]
    lines += [
        "", f"Limit: {100 * lim['lattice_volume']:g} %. With the mean of 2 x 2 x 2 samples per voxel every shape meets it. With the value at the voxel centre "
        "(the default, eight times cheaper) the balls meet it, and a cylinder whose flat ends are parallel to the lattice planes does not: an end "
        f"is located only to within the spacing ({100 * max(abs(r['rel_soft']) for r in point if 'ball' not in r['shape']):.1f} % here). "
        "`lattice(..., supersample=(2, 1, 1))` samples twice along z only.", "",
        "**The vial section against the 2D phantom.** The reference plane of `qa3d` (axial, through the isocentre) was compared with the 2D `qa` "
        f"world of the same seed and spec as `anatomy.qa` builds it, class by class on the master grid: {m['flat_2d']['cases']} cases "
        f"({len(m['seeds'])} seeds; the default spec, two oil vials, an elongated body; upright and turned by 30 degrees). Largest difference "
        f"{m['flat_2d']['max']:g}; equal bit for bit: {'yes' if m['flat_2d']['bitwise'] else 'no'}; the 3D classes absent from the plane, the tissue values and the "
        f"regions equal to the 2D ones: {'yes' if m['flat_2d']['all_else'] else 'no'}. Limit {lim['flat_2d']:g}.", "",
        "**Threads.** Worlds of both kinds (with and without options, two half-widths of the soft edge, 2D worlds between them) built serially "
        "and then from four threads in shuffled orders have the same sha256 over their planes, lattice labels, layout, regions and tissue values "
        "(`tests/test_volume_world.py`).", "",
        "## 6. Not verified", "",
        "- No image exists of a volume world: nothing here was compared with a scan, and no number of this page is a property of an image.",
        "- The anatomy was looked at on label maps of a few seeds; its realism is not measured.",
        "- Times were taken on one shared machine; the other numbers on one numerical stack. On another NumPy build the bit-for-bit statements "
        "hold only as far as that build rounds elementary functions the same way.",
        "- The susceptibilities other than air are declared, and nobody has signed them.", ""]
    return "\n".join(lines)


def paths():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(root, "docs", "physics", "volume.json"), os.path.join(root, "docs", "physics", "volume.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="measure, store docs/physics/volume.json and write docs/physics/volume.md")
    ap.add_argument("--doc", action="store_true", help="write the page from the stored measurement")
    a = ap.parse_args(argv)
    store, page = paths()
    if a.doc:
        with open(store, encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = measure()
        if not a.write:
            print(json.dumps(m, indent=1))
            return 0
        with open(store, "w", encoding="utf-8") as f:
            f.write(json.dumps(m, indent=1, sort_keys=True) + "\n")
        with open(store, encoding="utf-8") as f:                        # the page is written from what is stored
            m = json.load(f)
    with open(page, "w", encoding="utf-8") as f:
        f.write(doc_page(m))
    print("wrote", page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
