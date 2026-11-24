"""One episode with a scripted policy, through the public API only: look once, then prescribe the scan."""
import mrgym

env = mrgym.make("Scanner/NoWrap-v0", level=2, seed=0)
obs, info = env.reset()                                    # the task statement; a new hidden object
env.step({"name": "scan", "arguments": {"protocol": {      # a localiser: costs scan time
    "sequence": "spgr", "tr_ms": 10, "flip_deg": 10, "fov_mm": 288, "matrix": 160, "averages": 2}}})
box = env.step({"name": "view", "arguments": {"scan": 0}}).observation.data["bounding_box"]
half_y, half_x = (max(abs(v) for v in box[k]) + 1 for k in ("y_mm", "x_mm"))
pe, half_pe, half_ro = ("y", half_y, half_x) if half_y <= half_x else ("x", half_x, half_y)
lines, matrix = int(2 * -(-half_pe // 2)), int(2 * -(-half_ro // 2))          # 2 mm pixels; phase-encode the short side
obs, reward, terminated, truncated, info = env.step({"name": "submit", "arguments": {
    "pe_dir": pe, "fov_mm": 2.0 * matrix, "phase_fov": lines / matrix, "matrix": matrix}})
print(f"reward {reward:.2f} · scan time used {info['budget']['used_s']:.1f} s · delivered scan {info['score']['scan_s']:.1f} s "
      f"(fastest valid {info['score']['scan_s_fastest']:.1f} s, safe {info['score']['scan_s_safe']:.1f} s)")
