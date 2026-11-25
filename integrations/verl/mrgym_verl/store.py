"""Where the files of a run leave the node: ``s3://bucket/prefix`` (boto3, the node's role) or a directory.

    python -m mrgym_verl.store put LOCAL URI          one file
    python -m mrgym_verl.store get URI LOCAL          one file; exit 1 when it does not exist
    python -m mrgym_verl.store exists URI
    python -m mrgym_verl.store ls URI
    python -m mrgym_verl.store down URI DIR           every object under URI that DIR does not have yet
    python -m mrgym_verl.store up DIR URI             every file of DIR (new or changed since the last pass)
    python -m mrgym_verl.store loop DIR URI [SECONDS] ``up`` every SECONDS (45) until killed

The directory form is what the local simulation uses in place of a bucket: the same code path, another scheme.
Nothing is ever deleted at the destination (the job role cannot delete; a failed run needs a new run id).
Files ending in ``.tmp`` (and ``.tmp<pid>``) are never uploaded: writers here write a temporary file and rename it.

Checkpoints: ``publish_checkpoint`` copies a finished ``global_step_N`` folder and only then writes ``LATEST``;
``restore_checkpoint`` reads ``LATEST`` and pulls that folder. The earlier track lost hours to a checkpoint that
existed only on a preempted node.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time

_TMP = re.compile(r"\.tmp\d*$")


class Store:
    def __init__(self, uri: str):
        self.uri = uri.rstrip("/")
        self.s3 = self.uri.startswith("s3://")
        if self.s3:
            import boto3
            self.bucket, _, self.prefix = self.uri[5:].partition("/")
            self.client = boto3.client("s3")
        else:
            self.root = os.path.abspath(self.uri)

    def _key(self, rel: str) -> str:
        rel = rel.strip("/")
        if self.s3:
            return "/".join(x for x in (self.prefix, rel) if x)
        return os.path.join(self.root, rel) if rel else self.root

    def put(self, local: str, rel: str) -> None:
        if self.s3:
            self.client.upload_file(local, self.bucket, self._key(rel))
        else:
            dst = self._key(rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst + ".part")
            os.replace(dst + ".part", dst)

    def get(self, rel: str, local: str) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        if self.s3:
            try:
                self.client.download_file(self.bucket, self._key(rel), local)
                return True
            except Exception as e:  # noqa: BLE001  botocore.exceptions.ClientError 404
                if "404" in str(e) or "Not Found" in str(e) or "NoSuchKey" in str(e):
                    return False
                raise
        src = self._key(rel)
        if not os.path.isfile(src):
            return False
        shutil.copyfile(src, local)
        return True

    def exists(self, rel: str) -> bool:
        if self.s3:
            r = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self._key(rel), MaxKeys=1)
            return any(o["Key"] == self._key(rel) for o in r.get("Contents", []))
        return os.path.isfile(self._key(rel))

    def ls(self, rel: str = "") -> list[str]:
        """Relative keys of every object under ``rel``."""
        if self.s3:
            base, out, token = self._key(rel), [], None
            base = base + "/" if base and not base.endswith("/") else base
            while True:
                kw = dict(Bucket=self.bucket, Prefix=base)
                if token:
                    kw["ContinuationToken"] = token
                r = self.client.list_objects_v2(**kw)
                out += [o["Key"][len(base):] for o in r.get("Contents", []) if not o["Key"].endswith("/")]
                token = r.get("NextContinuationToken")
                if not token:
                    return sorted(out)
        base, out = self._key(rel), []
        for root, _, files in os.walk(base):
            out += [os.path.relpath(os.path.join(root, f), base) for f in files if not f.endswith(".part")]
        return sorted(out)

    def put_text(self, rel: str, text: str) -> None:
        if self.s3:
            self.client.put_object(Bucket=self.bucket, Key=self._key(rel), Body=text.encode())
        else:
            dst = self._key(rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".part", "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(dst + ".part", dst)

    def get_text(self, rel: str) -> str | None:
        if self.s3:
            try:
                return self.client.get_object(Bucket=self.bucket, Key=self._key(rel))["Body"].read().decode()
            except Exception as e:  # noqa: BLE001
                if "NoSuchKey" in str(e) or "404" in str(e):
                    return None
                raise
        try:
            with open(self._key(rel), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None


def down(store: Store, local_dir: str, rel: str = "") -> int:
    n = 0
    for key in store.ls(rel):
        dst = os.path.join(local_dir, key)
        if not os.path.exists(dst):
            store.get("/".join(x for x in (rel.strip("/"), key) if x), dst)
            n += 1
    return n


def up(store: Store, local_dir: str, rel: str = "", seen: dict | None = None) -> int:
    seen, n = {} if seen is None else seen, 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            path = os.path.join(root, f)
            if _TMP.search(f):
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            if seen.get(path) != sig:
                try:
                    store.put(path, "/".join(x for x in (rel.strip("/"), os.path.relpath(path, local_dir)) if x))
                    seen[path] = sig
                    n += 1
                except Exception as e:  # noqa: BLE001  the next pass tries again
                    print("upload failed", path, e, flush=True)
    return n


# ------------------------------------------------------------------ checkpoints
def publish_checkpoint(checkpoint_dir: str, dest_uri: str, global_step: int, extra: dict | None = None) -> dict:
    name = os.path.basename(os.path.normpath(checkpoint_dir))              # global_step_N
    store = Store(dest_uri)
    n = up(store, checkpoint_dir, name)
    latest = dict(global_step=int(global_step), folder=name, files=n, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {}))
    store.put_text("LATEST", json.dumps(latest))
    return latest


def latest_checkpoint(dest_uri: str) -> dict | None:
    text = Store(dest_uri).get_text("LATEST")
    return json.loads(text) if text else None


def restore_checkpoint(dest_uri: str, ckpt_dir: str) -> int | None:
    """Pull the newest published checkpoint into ``ckpt_dir`` and write verl's tracker file. Returns its step, or None."""
    latest = latest_checkpoint(dest_uri)
    if not latest:
        return None
    store, folder = Store(dest_uri), latest["folder"]
    got = down(store, os.path.join(ckpt_dir, folder), folder)
    if not os.path.isdir(os.path.join(ckpt_dir, folder)):
        raise RuntimeError(f"LATEST names {folder} but {dest_uri}/{folder} holds no file")
    with open(os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt"), "w", encoding="utf-8") as f:
        f.write(str(latest["global_step"]))
    print(f"[mrgym] restored {folder} from {dest_uri} ({got} file(s) pulled)", flush=True)
    return int(latest["global_step"])


def _split_file(uri: str) -> tuple[Store, str]:
    head, _, name = uri.rstrip("/").rpartition("/")
    return Store(head), name


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = argv[0], argv[1:]
    if cmd == "put":
        s, name = _split_file(rest[1])
        s.put(rest[0], name)
    elif cmd == "get":
        s, name = _split_file(rest[0])
        return 0 if s.get(name, rest[1]) else 1
    elif cmd == "exists":
        s, name = _split_file(rest[0])
        return 0 if s.exists(name) else 1
    elif cmd == "ls":
        print("\n".join(Store(rest[0]).ls()))
    elif cmd == "down":
        print("downloaded", down(Store(rest[0]), rest[1]), flush=True)
    elif cmd == "up":
        print("uploaded", up(Store(rest[1]), rest[0]), flush=True)
    elif cmd == "loop":
        store, seen, every = Store(rest[1]), {}, float(rest[2]) if len(rest) > 2 else 45.0
        while True:
            up(store, rest[0], "", seen)
            time.sleep(every)
    elif cmd == "restore":
        step = restore_checkpoint(rest[0], rest[1])
        print("" if step is None else step)
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
