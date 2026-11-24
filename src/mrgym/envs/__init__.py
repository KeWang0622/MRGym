"""Built-in environments. Importing this module registers their ids."""
from ..registry import register

register("Scanner/Explore-v0", "mrgym.envs.explore:ExploreEnv", "mrgym.envs.explore:card", kind="sandbox",
         levels=(1,), default_level=1, max_steps=1024, reward_range=(0.0, 0.0), reward_threshold=0.0)
register("Scanner/NoWrap-v0", "mrgym.envs.nowrap.env:NoWrapEnv", "mrgym.envs.nowrap.env:card", kind="admitted",
         family="mrgym.envs.nowrap", levels=(1, 2), default_level=1, max_steps=40)
register("Scanner/T1Transmit-v0", "mrgym.envs.t1_transmit.env:T1TransmitEnv", "mrgym.envs.t1_transmit.env:card", kind="practice",
         family="mrgym.envs.t1_transmit", levels=(1, 2, 3), default_level=2, max_steps=64)
