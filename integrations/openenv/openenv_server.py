"""Compatibility shim: the OpenEnv face of MR Gym now lives in ``mrgym_scanner_env/server/mrgym_environment.py``.

    from openenv_server import make_app, make_environment_class
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrgym_scanner_env"))

from server.mrgym_environment import MRGymState, make_app, make_environment_class  # noqa: E402

__all__ = ["MRGymState", "make_app", "make_environment_class"]
