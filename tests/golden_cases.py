"""What the golden files hold and how it is computed. Used by ``tests/test_golden.py``, ``tests/test_task_snapshots.py``
and by ``tools/audit/golden.py``, which writes the files (nothing under ``tests/golden/`` is typed by hand).

``engine_digests.json``   a function of the engine and the numerical stack only
    engine    ``mrgym.ENGINE_VERSION`` the numbers were recorded on
    stack     numpy, scipy, machine, the SIMD extensions numpy found: the reference stack (bit-exact there)
    images    6 sequences x 3 phantoms x 2 field strengths, on one phantom the variants of engine 0.2.0 (hard pulses,
              other pulse shapes, a turned field of view), and on a phantom with oil vials the variants of engine 0.3.0
              (fat saturation of the six-line fat, partial Fourier, the adjustments): the noise-free image (sha256 of
              the float32 magnitude and phase, the 8 x 8 block fingerprint of the magnitude, the shape)
    noisy     one session: every sequence once with noise, one protocol twice (sha256, fingerprint), the chain head
    prices    lines and scan seconds of a grid of protocols (console arithmetic: equal on every stack)
    array     additive: a world with a receive array: its maps and noise covariance, coil images of both formations, a
              noisy accelerated scan through SENSE, the g map (``array_cases``)
``volume_digests.json``   the volume worlds (no scan): see ``VOLUME_WORLDS`` below
``compat_2d.json``        the 2D baseline recorded on v1: ``tests/compat_cases.py`` (checked here, written only there)
``task_snapshots.json``   a function of the engine AND of the task and control code of each id
    per id    instance (seed 0) and statement per level, the submit schema, the card, the module constants, and one
              episode of the reference policy (sha256 of every observation text, chain head, reward)
    shared    the constants of the gate, of ``family.py``, of the log and the console's cost table

Rules (CONTRIBUTING.md, "Golden data"): a change of ``images`` / ``noisy`` / ``prices`` needs a new ``ENGINE_VERSION``;
a change of a task snapshot of a released id needs a new ``-vN``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re

import numpy as np
import scipy

import mrgym
from mrgym import gate, policies
from mrgym.env import derive_seed
from mrgym.envs import family
from mrgym.scanner import console, kspace, session
from mrgym.scanner import world as W

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
ENGINE_FILE = os.path.join(HERE, "engine_digests.json")
TASK_FILE = os.path.join(HERE, "task_snapshots.json")
VOLUME_FILE = os.path.join(HERE, "volume_digests.json")
SEED = "000102030405060708090a0b0c0d0e0f"
GRID = 320                                  # master grid of the image cases (the engine is the same code at any grid)
PHANTOMS, FIELDS = ("qa", "brain", "abdomen"), (1.5, 3.0)
PROTOCOLS = dict(
    spgr=dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, n_echoes=2, ro_polarity=-1),
    afi=dict(sequence="afi", tr_ms=25.0, flip_deg=55.0, matrix=64),
    se=dict(sequence="se", tr_ms=500.0, matrix=64, pe_dir="x", centre_x_mm=12.0, phase_fov=0.75),
    ir=dict(sequence="ir", tr_ms=1500.0, ti_ms=400.0, matrix=64, fat_sat=True),
    bssfp=dict(sequence="bssfp", tr_ms=8.0, flip_deg=30.0, matrix=64, phase_oversampling=0.25),
    epi=dict(sequence="epi", tr_ms=200.0, matrix=64, partial_fourier=0.75),
)
# What engine 0.2.0 added, on the QA phantom at 3 T: every sequence with hard pulses (``excitation: nonselective``, the
# 3D path), the two other pulse shapes, a turned field of view, a shifted fat-saturation pulse.
VARIANTS = dict(
    **{f"{k}+nonselective": dict(v, excitation="nonselective") for k, v in PROTOCOLS.items()},
    **{"spgr+sharp": dict(PROTOCOLS["spgr"], rf_pulse="sharp"), "se+fast": dict(PROTOCOLS["se"], rf_pulse="fast"),
       "se+rotated": dict(PROTOCOLS["se"], rotation_deg=30.0), "ir+fatsat_offset": dict(PROTOCOLS["ir"], fatsat_offset_hz=40.0)})
# What engine 0.3.0 changed or added, on a QA phantom with three oil vials at 3 T (``qa_oil``): fat saturation of a fat
# with six lines (with hard pulses: nothing but the saturation differs from the unsaturated scan), partial Fourier
# reconstructed by homodyne detection, the four adjustment parameters, and a `bssfp` phase cycle that is neither 0 nor 180 degrees.
WORLD_SPECS = dict(qa_oil=dict(kind="qa", qa=dict(n_oil=3)))
VARIANTS_0_3 = {
    "se+fat_sat+nonselective": dict(sequence="se", tr_ms=500.0, matrix=64, fat_sat=True, excitation="nonselective"),
    "spgr+fat_sat": dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, fat_sat=True),
    "epi+fat_sat": dict(sequence="epi", tr_ms=200.0, matrix=64),
    "se+partial_fourier+nonselective": dict(sequence="se", tr_ms=500.0, matrix=64, partial_fourier=0.6, excitation="nonselective"),
    "spgr+adjusted": dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, cf_shift_hz=40.0, tx_scale=1.15, shim_y_hz_per_mm=0.2,
                          shim_x_hz_per_mm=-0.1),
    "bssfp+adjusted": dict(sequence="bssfp", tr_ms=8.0, flip_deg=30.0, matrix=64, cf_shift_hz=-25.0, shim_x_hz_per_mm=0.3),
    "bssfp+phase_cycle": dict(sequence="bssfp", tr_ms=8.0, flip_deg=30.0, matrix=64, phase_cycle_deg=90.0),
    "bssfp+phase_cycle+nonselective": dict(sequence="bssfp", tr_ms=8.0, flip_deg=30.0, matrix=64, phase_cycle_deg=90.0, excitation="nonselective"),
}
PRICE_GRID = [dict(sequence=s, **geo) for s in ("se", "spgr", "epi")
              for geo in (dict(), dict(partial_fourier=0.55), dict(partial_fourier=0.75), dict(phase_oversampling=0.5),
                          dict(averages=4), dict(phase_fov=0.5, partial_fourier=0.75, phase_oversampling=0.5, averages=4),
                          dict(matrix=64), dict(excitation="nonselective"), dict(excitation="nonselective", slice_mm=10.0),
                          dict(rf_pulse="sharp"))]
PRICE_BASE = dict(se=dict(tr_ms=500.0), spgr=dict(tr_ms=20.0, flip_deg=15.0), epi=dict(tr_ms=200.0))


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def jsonable(obj):
    """Tuples become lists and numpy scalars Python numbers, as a JSON round trip leaves them."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else list(o)))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- the stack
def stack() -> dict:
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:                                                 # numpy 1.x
        from numpy.core._multiarray_umath import __cpu_features__ as feats
    return dict(numpy=np.__version__, scipy=scipy.__version__, machine=platform.machine(),
                simd=sorted(k for k, v in feats.items() if v))


def mode(recorded: dict) -> tuple[str, str]:
    """("bit-exact" | "tolerance", why). Bit for bit only on the stack the digests were recorded on."""
    forced = os.environ.get("MRGYM_GOLDEN_MODE")
    if forced in ("bit-exact", "tolerance"):
        return forced, "MRGYM_GOLDEN_MODE"
    now = stack()
    diff = [k for k in ("numpy", "scipy", "machine", "simd") if now[k] != recorded.get(k)]
    if diff:
        return "tolerance", "this stack differs from the recorded one in " + ", ".join(diff)
    return "bit-exact", f"numpy {now['numpy']}, scipy {now['scipy']}, {now['machine']}: the recorded stack"


# ---------------------------------------------------------------------------------------------- engine digests
def _digest(images) -> dict:
    mag, ph = np.abs(images).astype(np.float32), np.angle(images).astype(np.float32)
    return dict(shape=list(images.shape), sha256=hashlib.sha256(mag.tobytes() + ph.tobytes()).hexdigest(),
                fingerprint=session.fingerprint(mag))


def image_keys() -> list[str]:
    return ([f"{p}/{b}T/{s}" for p in PHANTOMS for b in FIELDS for s in PROTOCOLS] + [f"qa/3.0T/{v}" for v in VARIANTS]
            + [f"qa_oil/3.0T/{v}" for v in VARIANTS_0_3])


_ENGINES: dict = {}


def image_case(key: str) -> dict:
    phantom, field, seq = key.split("/")
    k = (phantom, field)
    if k not in _ENGINES:
        _ENGINES.clear()                                                # one world at a time: the keys come grouped
        _ENGINES[k] = kspace.Engine(W.make_world(SEED, dict(WORLD_SPECS.get(phantom, dict(kind=phantom)), b0_t=float(field[:-1]), n=GRID)))
    protocol = VARIANTS_0_3[seq] if phantom in WORLD_SPECS else PROTOCOLS[seq] if seq in PROTOCOLS else VARIANTS[seq]
    return _digest(_ENGINES[k].scan(protocol)["images"])


def noisy_session() -> dict:
    cfg = dict(task_id="golden", seed=SEED, world_spec=dict(kind="qa", n=GRID), budget_s=1e5, max_requests=32, task=dict(regions=[]))
    s = session.ScannerSession(cfg)
    out = {}
    for name in [*PROTOCOLS, "se"]:                                      # the second se scan draws other noise
        s.scan({"protocol": PROTOCOLS[name]})
        ev = s.events[-1]
        out[f"{len(out)}:{name}"] = dict(shape=ev["shape"], sha256=ev["data_sha256"], fingerprint=ev["fingerprint"])
    out["chain_head"] = s.chain
    return out


def price_key(p: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in p.items())


def prices() -> dict:
    out = {}
    for p in PRICE_GRID:
        pl = console.plan(dict(PRICE_BASE[p["sequence"]], **p), dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5))
        out[price_key(p)] = dict(lines=pl["realised"]["lines"], lines_full=pl["realised"]["lines_full"], scan_s=pl["scan_s"],
                                 parts=pl["scan_s_parts"])
    return out


def array_cases() -> dict:
    """The additive section ``array``: a world with a receive array (QA phantom, 4 loops on a ring). The declared
    array itself (maps on a lattice, psi), the coil images of the fast and of the per-coil formation, a noisy scan
    that keeps every second line reconstructed with SENSE and the true maps, its g map. Nothing above depends on it."""
    from mrgym.scanner import array as A
    from mrgym.scanner import recon
    w = W.make_world(SEED, dict(kind="qa", n=GRID, array=dict(n=4)))
    ae, arr = A.ArrayEngine(w), w["array"]
    c = np.linspace(-80.0, 80.0, 16)
    out = dict(maps=_digest(arr["maps"](c[:, None], c[None, :])),
               psi=[[round(float(v.real), 9), round(float(v.imag), 9)] for v in np.asarray(arr["psi"]).ravel()])
    rp = console.plan(PROTOCOLS["spgr"], w["console"])["realised"]
    out["fast/spgr"] = _digest(ae.coil_images(ae.clean_kspace(rp, "fast"), rp).reshape(-1, rp["matrix_pe"], rp["matrix"]))
    rp_se = console.plan(PROTOCOLS["se"], w["console"])["realised"]
    out["per_coil/se"] = _digest(ae.coil_images(ae.clean_kspace(rp_se, "per_coil"), rp_se)[0])
    lines = recon.sampling_mask(rp["lines_full"], 2)
    ks = ae.acquire_kspace(rp, np.random.default_rng(np.random.SeedSequence(entropy=W.seed_int(SEED), spawn_key=(W.STREAMS["noise"], 0, 0))), lines)
    plan = recon.sense_plan(ae.pixel_maps(rp), ae.psi, 2)
    out["sense_r2/spgr"] = _digest(np.array([ae.to_image(recon.sense_unfold(plan, ae.coil_images(k, rp, full_pe=True)), rp) for k in ks]))
    out["g_r2"] = _digest(ae.to_image(np.nan_to_num(plan["g"]), rp)[None])
    return out


def engine_digests() -> dict:
    return dict(engine=mrgym.ENGINE_VERSION, stack=stack(), images={k: image_case(k) for k in image_keys()},
                noisy=noisy_session(), prices=prices(), array=array_cases())


# ---------------------------------------------------------------------------------------------- volume worlds
# ``volume_digests.json``: the volume worlds (``mrgym.scanner.volume``) as they are, an additive file: per case the sha256
# of ``volume_check.digest`` (reference plane, two other planes, lattice labels, layout, regions, tissues, system) and,
# for other numerical stacks, the volume of every class on an 8 mm lattice. No scan is in it.
VOLUME_WORLDS = {
    "head3d/3.0T/lesions": dict(kind="head3d", lesion="ms"),
    "head3d/1.5T/no vessels": dict(kind="head3d", b0_t=1.5, volume=dict(vessels=0)),
    "qa3d/3.0T": dict(kind="qa3d"),
    "qa3d/3.0T/oil, posed": dict(kind="qa3d", qa=dict(n_oil=1), volume=dict(pose_deg=5.0)),
}


def volume_case(key: str) -> dict:
    from mrgym.scanner import volume_check
    w = W.make_world(SEED, VOLUME_WORLDS[key])
    vol = w["volume"]
    lat = vol.lattice(8.0)
    return dict(sha256=volume_check.digest(w), classes=list(vol.matter), regions=len(vol.regions),
                class_mm3=[round(float(v) * 8.0 ** 3, 1) for v in lat.weights.sum((1, 2, 3), dtype=np.float64)])


def volume_digests() -> dict:
    return dict(stack=stack(), worlds={k: volume_case(k) for k in VOLUME_WORLDS})


# ---------------------------------------------------------------------------------------------- task snapshots
def task_ids() -> list[str]:
    return sorted(i for i, s in mrgym.registry.items() if s.family and s.family.startswith("mrgym."))


def _constants(module) -> dict:
    out = {}
    for name in sorted(vars(module)):
        if name.isupper() and not name.startswith("_"):
            try:
                out[name] = jsonable(getattr(module, name))
            except TypeError:
                pass
    return out


def task_static(env_id: str) -> dict:
    """Everything of an id that does not need an episode: instances, texts, schema, card, constants."""
    spec = mrgym.spec(env_id)
    task = importlib.import_module(spec.family + ".task")
    out = dict(levels={}, constants=_constants(task))
    for level in spec.levels:
        cfg = task.make_instance(derive_seed(env_id, level, "train", 0, environ={}), level, task_id="snapshot")
        row = dict(instance=jsonable(cfg), statement=task.statement(cfg))
        if hasattr(task, "submit_schema"):
            row["submit_schema"] = jsonable(task.submit_schema(cfg))
        if hasattr(task, "hint_text"):
            row["hints"] = task.hint_text(cfg)                           # the hint tiers of the level, without acquired scans
        out["levels"][str(level)] = row
    if hasattr(task, "card"):
        out["card"] = jsonable(task.card())
    if hasattr(task, "SUBMIT_DOC"):
        out["submit_doc"] = task.SUBMIT_DOC
    return out


def episode(env_id: str) -> dict:
    """One episode of the reference policy on seed 0 of the default level."""
    controls = policies.controls_of(mrgym.spec(env_id))
    env = mrgym.make(env_id, seed=0)
    res = policies.run(env, getattr(controls, "REFERENCE", "reference"))
    texts = [e["text"] for e in env.transcript()]
    log = env.scanner_log()
    out = dict(reward=res.reward, passed=bool(res.info.get("score", {}).get("passed")), steps=len(texts) - 1,
               used_s=res.info.get("budget", {}).get("used_s"), texts_sha256=sha(canon(texts)),
               chain_head=log[-1]["chain"] if log else None, policy_error=res.info.get("policy_error"))
    env.close()
    return out


def shared_constants() -> dict:
    return dict(gate=dict(ALPHA=gate.ALPHA, REF_MIN=gate.REF_MIN, BLIND_MAX=gate.BLIND_MAX, SEP_MIN=gate.SEP_MIN,
                          TUNING_SETS=gate.TUNING_SETS, TUNING_WORLDS=gate.TUNING_WORLDS, SCHEMA=gate.SCHEMA),
                family=dict(GATES=list(family.GATES)),
                log=dict(LOG_VERSION=session.LOG_VERSION, REPLAY_RTOL=session.REPLAY_RTOL, MAX_ROIS=session.MAX_ROIS,
                         MAX_POSITION_MM=session.MAX_POSITION_MM, FINGERPRINT_BLOCKS=session.FINGERPRINT_BLOCKS),
                console=jsonable(console.SYSTEM))


def task_snapshots() -> dict:
    ids = {}
    for env_id in task_ids():
        try:
            ids[env_id] = dict(task_static(env_id), episode=episode(env_id))
        except Exception as e:  # noqa: BLE001  a family that cannot be snapshotted is a named entry, not a lost file
            ids[env_id] = dict(error=f"{type(e).__name__}: {e}"[:300])
    return dict(engine=mrgym.ENGINE_VERSION, shared=shared_constants(), ids=ids)


def load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def dump(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True)
    # lists of plain values on one line (a fingerprint is 64 numbers): the file stays readable in a diff
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "]", text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
