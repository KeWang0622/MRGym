"""MR Gym: a virtual MRI scanner you can drop an agent into.

    import mrgym
    env = mrgym.make("Scanner/NoWrap-v1", level=1, seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step({"name": "plan", "arguments": {"protocol": {...}}})

Three versions travel with every number: the package (``mrgym.__version__``), the scanner engine
(``mrgym.ENGINE_ID``: the same seed and protocol give the same data only on the same engine), and the environment
id suffix (``-v1``: rewards are comparable only within one version).
"""
import os as _os

# A scan is single-threaded work; BLAS thread pools only add contention, most of all when episodes run in parallel.
# Set before numpy loads, and only if the caller has not chosen a value.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from . import envs as _envs  # noqa: E402, F401  registers the built-in environments
from .env import Env, Observation, ResetNeeded, StepResult  # noqa: E402
from .registry import EnvSpec, Error, RetiredEnv, UnknownEnv, card, list_envs, make, register, registry, spec  # noqa: E402
from .scanner.session import ENGINE_ID, ENGINE_VERSION  # noqa: E402
from .tools import ToolSpec  # noqa: E402

__version__ = "0.1.0.dev0"
__all__ = ["ENGINE_ID", "ENGINE_VERSION", "Env", "EnvSpec", "Error", "Observation", "ResetNeeded", "RetiredEnv", "StepResult", "ToolSpec",
           "UnknownEnv", "card", "list_envs", "make", "register", "registry", "spec", "__version__"]
