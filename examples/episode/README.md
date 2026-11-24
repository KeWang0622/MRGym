# A recorded episode

`episode_bundle.json` is one episode of `Scanner/NoWrap-v0` (level 2, seed 0) played by the scripted `reference`
policy: every tool call and reply, the localiser image as float32, the scanner's log, the verifier's result and,
under the key `hidden`, the world the agent never saw. No language model was involved. The format is described in
`src/mrgym/record.py`.

`viewer.html` replays it in a browser, next to the `size_only` policy on the same hidden world, which measures the
object's size, ignores that it lies off-centre and fails. Open the file from disk: it is self-contained and makes no
network request. The reward, the gates and the hidden maps appear at the last step.

Regenerate both files from the repository root:

    mrgym record Scanner/NoWrap-v0 --policy reference --level 2 --seed 0 --out examples/episode
    mrgym record Scanner/NoWrap-v0 --policy size_only --level 2 --seed 0 --out runs/size_only && mrgym view examples/episode runs/size_only --out examples/episode/viewer.html

The second line records the failing episode into `runs/` (not committed) and builds the viewer from the two bundles.
A new recording differs from the committed one only in the wall-clock stamps of the scanner log.

To read the bundle in Python:

    from mrgym import record
    bundle = record.load("examples/episode")
    magnitude, phase = record.scan_arrays(bundle, 0)      # float32 [1, 160, 160]
    pd = record.hidden_map(bundle, "pd")                  # what the agent never saw
