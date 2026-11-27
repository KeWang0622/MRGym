# Textbook experiments on the virtual scanner

Thirteen classic MR experiments, one script each. Every script prints a few lines of numbers and asserts the
textbook result with a stated tolerance. A failed assertion ends the script with a non-zero exit code.

```bash
pip install -e .
python examples/textbook/01_ernst_angle.py
python examples/textbook/01_ernst_angle.py --plot ernst.png     # optional figure, needs matplotlib
pytest tests/test_textbook_examples.py                          # all thirteen
```

The scripts need numpy and the package. No file is written unless `--plot` is given. Each one runs in under
15 s on one core.

## How the scripts reach the scanner

Scripts 01 to 04, 06, 08 and 13 open a session the way an agent does: `mrgym.make("Scanner/Explore-v0", phantom="qa",
seed=1)`, then `scan`, `read_regions` and `view` tool calls. They know their own seed, so they rebuild the hidden
world with `env.episode_config()` and compare with it. An agent in an episode cannot do that.

Scripts 05, 07, 09, 10 and 11 need an object and a field that are known exactly. They build a small world with
`world.shapes_phantom` and `world.flat_world` and image it with `kspace.Engine`, without noise. Script 01 also images
its session's world with the ideal-spoiling engine (`tier=1`). Script 12 uses the Bloch chain in
`mrgym.scanner.bloch`; it runs no scan.

`_common.py` holds the shared helpers: session, tool call, read-out columns, sinc interpolation, `check`.

## The experiments

| # | Experiment | Command | Expected result (asserted) | What it prints |
|---|---|---|---|---|
| 01 | Ernst angle | `python examples/textbook/01_ernst_angle.py` | Ideal spoiling: peak at arccos(exp(-TR / T1)) / B1 within 0.15 deg. Served: peak at the maximum of the RF-spoiled steady state within 0.4 deg. The two tiers differ by more than 5 % somewhere on the grid. | Per vial: T1, B1, Ernst angle / B1, both measured peaks. Range of served / Ernst signal over tissues, flip angles, TR and spoiler moments. Spoiler lobe duration. |
| 02 | Inversion recovery | `python examples/textbook/02_inversion_recovery.py` | T1 from a three-parameter fit within 1 %. Null point at T1 ln(1 + kappa) within 1 %. A scan at the null point leaves under 2 % of the recovered signal. | Header of one scan. Per vial: fitted and true T1, null point, T1 ln 2, inversion efficiency. |
| 03 | T2 from spin echoes | `python examples/textbook/03_t2_spin_echo.py` | T2 from a log-linear fit over six TE within 2 %. | Header of one scan. Per vial: fitted and true T2, and its T2*. Scan time of the series. |
| 04 | T2* and field map | `python examples/textbook/04_t2star_field_map.py` | T2* from the magnitude mean within 2 %. Off-resonance from the phase slope within 0.5 Hz, positive slope for positive off-resonance. The complex mean decays faster in every vial, by more than 5 % in at least one. | Header. Per vial: both T2* estimates, fitted and true off-resonance, field spread across the region. |
| 05 | Chemical shift | `python examples/textbook/05_chemical_shift.py` | Main fat peak moves by -polarity x f_fat / bandwidth pixels within 0.03 px, towards +readout for polarity +1. Header figure agrees within 0.001 px. Six-peak fat moves by its amplitude-weighted mean shift within 0.05 px. Water and the phase axis: no shift within 0.01 px. | Shift in pixels at four bandwidths, 1.5 and 3 T, for one peak and for six peaks. |
| 06 | Wrap-around | `python examples/textbook/06_wrap_around.py` | Folded read-out = region + region one field of view away, within 3 % of the largest signal. Phase oversampling removes the fold. The readout axis does not fold. The fold follows `pe_dir`. | Object extent against the truth. Size of the fold. One line per fold law. |
| 07 | Gibbs ringing | `python examples/textbook/07_gibbs_ringing.py` | Overshoot 8.95 % of the step within 0.3 at 4 mm pixels. First maximum one pixel inside the edge, ripple period two pixels, within 0.1 px. Overshoot falls at smaller pixels (master grid) and with a soft edge. | Per matrix: overshoot, first maximum, ripple period, overshoot with a 0.5 mm soft edge. |
| 08 | Noise | `python examples/textbook/08_noise.py` | Image noise scales with matrix^1.5, bandwidth^0.5, averages^-0.5, exponents within 0.05. Background magnitude: mean 1.2533 sigma, SD 0.6551 sigma, within 0.02. Magnitude mean of a weak signal follows the Rician expectation within 6 %. | Three exponents. Two Rayleigh ratios. Magnitude mean over complex mean for a weak signal. |
| 09 | bSSFP bands | `python examples/textbook/09_bssfp_bands.py` | Dark bands at off-resonance (m + 1/2) / TR within 1 mm in a 2 Hz/mm field. A phase cycle of 0 deg puts a band on resonance, one of 90 deg at +1 / (4 TR). | Band positions and spacing at TR 5, 8, 10 ms. |
| 10 | EPI distortion | `python examples/textbook/10_epi_distortion.py` | Shift -polarity x offset x echo spacing x FOV within 0.3 mm. Stretch 1 -/+ gradient x echo spacing x FOV within 0.01. Mean of blip-up and blip-down centres is the true centre within 0.3 mm. Brightness ratio within 3 %. | Per blip polarity: edges, centre, height. Mean centre. Brightness ratio. |
| 11 | In and opposed phase | `python examples/textbook/11_in_opposed_phase.py` | One fat peak: minima at (k + 1/2) / abs(f_fat) within 0.02 ms, maxima at k / abs(f_fat) within 0.1 ms and never later. Six peaks: minima within 0.2 ms of those times, and deeper. | Per field strength: echo times of minima and maxima for one and for six fat peaks. |
| 12 | Approach to steady state | `python examples/textbook/12_approach_to_steady_state.py` | First line after n dummy cycles = m (1 + (1 / m - 1) (E1 cos a)^n) within 1e-12. Spin echo with exact pulses: one dummy cycle reaches the periodic state. | Dummy cycles per TR. Per T1: first line and centre of k-space against the periodic state. Spin echo with a transmit error. |
| 13 | Slice profile | `python examples/textbook/13_slice_profile_vfa.py` | Two-angle T1 with the nominal angles on a 2D slice: inside -35 to -80 % for time-bandwidth products 2 and 4 (published: 40 to 80 % underestimation, Svedin & Parker 2017; -63 %, Dieringer 2014), less for 8. The same two scans in 3D with the true angles: within 3 %. The signal maximum of a flip-angle sweep lies above the Ernst angle in 2D. | Bias range per pulse. T1 from two scans in 2D and in 3D, with nominal and with true angles. Flip angle of maximum signal in both modes. |

Example 12 produces the transient sizes quoted in [docs/physics/limits.md](../../docs/physics/limits.md).
Example 01 produces the served / Ernst range quoted there, example 07 the overshoot at small pixels and example 13
the slice-profile bias.

Examples 01 and 12 test a formula for one flip angle per voxel. They ask for `excitation="nonselective"` (hard
pulses, 3D encoding); with the default slice-selective pulse the flip angle varies through the slice, which is what
example 13 is about.

## Reading the output

`check` prints one line per assertion: the measured value, the expected value and the tolerance. Tolerances are
wide enough for the noise of the fixed seed and narrow enough to catch a wrong sign, a wrong axis or a wrong
exponent. The seed is fixed, so two runs on one machine print the same numbers.
