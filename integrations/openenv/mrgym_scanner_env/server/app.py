"""FastAPI application of the MR Gym OpenEnv package: ``/ws``, ``/reset``, ``/step``, ``/state``, ``/schema``.

    uvicorn server.app:app --host 0.0.0.0 --port 8000

The default environment id of an episode is ``$MRGYM_ENV_ID``, else the id this package was exported for
(``default_env.json``, written by ``mrgym export openenv``). ``reset(env_id=...)`` selects any other registered id.
"""
import json
import os

try:
    from .mrgym_environment import make_app
except ImportError:                                       # run from the package directory: `uvicorn server.app:app`
    from server.mrgym_environment import make_app

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_env_id() -> str:
    if os.environ.get("MRGYM_ENV_ID"):
        return os.environ["MRGYM_ENV_ID"]
    with open(os.path.join(_HERE, "default_env.json"), encoding="utf-8") as f:
        return json.load(f)["env_id"]


app = make_app(default_env_id(), max_concurrent_envs=int(os.environ.get("MRGYM_MAX_SESSIONS", "8")))


def main():
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
