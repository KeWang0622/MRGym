"""One episode with a scripted policy, through the public API only: look once, measure, then prescribe the scan."""
import mrgym

env = mrgym.make("Scanner/NoWrap-v1", level=2, seed=0)     # level 2: the object is off-centre, its long axis lies along x or y
obs, info = env.reset()                                    # the task statement and its limits; a new hidden object
env.step({"name": "scan", "arguments": {                   # a localiser: costs scan time
    "sequence": "spgr", "tr_ms": 11, "flip_deg": 10, "fov_mm": 288, "matrix": 128, "averages": 2}})
edge = {}
for axis in ("y", "x"):                                    # the brightest pixel of every row, then of every column
    p = env.step({"name": "profile", "arguments": {"scan_index": 0, "axis": axis, "width_mm": 288, "reduce": "max"}}).observation.data
    inside = [pos for pos, value in zip(p["positions_mm"], p["values"]) if value > 0.3 * max(p["values"])]
    edge[axis] = (min(inside) - p["sample_mm"] / 2, max(inside) + p["sample_mm"] / 2)      # outer edges of the object
extent = {axis: high - low + 4 for axis, (low, high) in edge.items()}                      # 2 mm to spare on each side
pe, ro = ("y", "x") if extent["y"] <= extent["x"] else ("x", "y")                          # phase-encode the short side
lines, matrix = int(2 * -(-extent[pe] // 4)), max(int(2 * -(-extent[ro] // 4)), 40)        # 2 mm pixels
obs, reward, terminated, truncated, info = env.step({"name": "submit", "arguments": {
    "pe_dir": pe, "fov_mm": 2.0 * matrix, "phase_fov": lines / matrix, "matrix": matrix, "rotation_deg": 0,
    "centre_y_mm": sum(edge["y"]) / 2 + 1, "centre_x_mm": sum(edge["x"]) / 2 + 1}})        # the image reaches half a pixel less far on the high side
print(f"reward {reward:.0f} · scan time used {info['budget']['used_s']:.1f} s · delivered scan {info['score']['scan_s']:.1f} s "
      f"(fastest valid {info['score']['scan_s_fastest']:.1f} s)")
