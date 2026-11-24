# Figures

Every image under `assets/` is written by a script in this folder from a run of the scanner engine in this
repository. Nothing is drawn by hand except the loop diagram, whose labels name parts of the package.

Regenerate all of them (matplotlib is needed: `pip install -e ".[viz]"`):

    python tools/figures/make_all.py          # or: make figures

| Output | Command | What it shows |
|---|---|---|
| `assets/hero.png` | `python tools/figures/hero.py` | One scripted session on the brain phantom at 3 T through `Scanner/Explore-v0`: five requests, the images, the console header lines, the scan-time budget |
| `assets/episode_nowrap.png` | `python tools/figures/episode.py` | One recorded `Scanner/NoWrap-v0` level 2 episode of the reference policy next to the `size_only` policy on the same hidden world |
| `assets/gallery_sequences.png` | `python tools/figures/gallery_sequences.py` | One brain through spgr, se (T1 and T2 weighting), ir, bssfp and epi; the abdomen with and without fat saturation |
| `assets/gallery_physics.png` | `python tools/figures/gallery_physics.py` | Chemical shift against bandwidth and polarity, wrap-around, truncation ringing, bSSFP bands, EPI distortion |
| `assets/loop.png`, `assets/loop.svg` | `python tools/figures/loop.py` | Agent, tools, scanner (console and engine), hidden world, log, verifier and reward |
| `assets/hidden_vs_received.png` | `python tools/figures/hidden_vs_received.py` | Maps of the hidden world above everything one scan call returns |

Each script takes `--out PATH`. `style.py` holds the palette, the text sizes and the drawing helpers; `scenes.py`
holds the two gallery phantoms (the worlds `mrgym scan --seed 1` images).

Sizes: figures are 9.6 in wide and are meant to be shown 880 px wide. A PNG above 600 kB is written again at a lower
resolution until it fits, never below 120 dpi (1152 px wide).

## The episode figure from a language-model episode

`episode.py` draws from episode bundles (`mrgym record`, or `mrgym.record.bundle_from_env` in a model runner), so a
model episode can replace the scripted one without changing the script. Pass the model's bundle first and a failing
scripted episode of the same hidden world second; record the second with the level and seed the model played:

    mrgym record Scanner/NoWrap-v0 --policy size_only --level 2 --seed 0 --out runs/size_only
    python tools/figures/episode.py --bundle runs/model/episode_bundle.json runs/size_only/episode_bundle.json \
        --out assets/episode_nowrap_model.png

The figure labels each row with the agent recorded in its bundle ("scripted policy" or "language model") and states
"same hidden world" only when the two bundles carry the same task id.

## What is scripted

No figure here involves a language model. The sessions in `hero.py` and `hidden_vs_received.py` are lists of
requests written in those files and played through `env.step`; the episodes in `episode.py` are control policies of
the environment. Each figure says so in its own text.
