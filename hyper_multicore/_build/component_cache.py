# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Content-addressed native components, published only after the artifact transaction commits."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time

from .build_utils import atomic_json
from .command_runner import message


def sha256(path: Path) -> str:
    """Hash an input or output using bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(inputs: dict) -> str:
    """Fingerprint semantic inputs, excluding producer locations and timing."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def files_in(directory: Path) -> dict[str, str]:
    """Require exact, regular-file cache membership without following redirects."""
    result = {}
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Invalid component payload: {directory}")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Redirected or special component content: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha256(path)
    if not result:
        raise ValueError(f"Empty component payload: {directory}")
    return result


def _owned(directory: Path) -> None:
    """Never traverse another user's or a writable redirected cache directory."""
    for path in (directory, *directory.parents):
        if path.is_symlink():
            raise ValueError(f"Symlinked component cache: {path}")
    status = directory.stat()
    if status.st_uid != os.getuid() or status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"Component cache must be private to the current user: {directory}")


def cache_root() -> Path:
    """Locate deliberately shared, per-user storage without changing worktree ownership."""
    configured = os.environ.get("HYPER_MULTICORE_NATIVE_CACHE")
    root = Path(configured).expanduser().absolute() if configured else (
        Path.home() / ".cache/hyper-multicore/native-components-v1")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError(f"Symlinked component cache: {path}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _owned(root)
    marker = root / "owner.json"
    expected = {"schema": 1, "uid": os.getuid(), "kind": "native-components"}
    with _lock(root / ".owner.lock"):
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text()) != expected:
                raise ValueError(f"Foreign component cache: {root}")
        elif any(p.name != ".owner.lock" for p in root.iterdir()):
            raise ValueError(f"Unattributed component cache: {root}")
        else:
            atomic_json(marker, expected)
    return root


@contextmanager
def _lock(path: Path):
    """Keep a persistent inode so simultaneous publishers cannot replace each other's lock."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _entry(root: Path, name: str, key: str) -> Path:
    if (name in {".", ".."} or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name)
            or not re.fullmatch(r"[a-f0-9]{64}", key)):
        raise ValueError("Invalid component cache identity")
    directory = root / name
    directory.mkdir(exist_ok=True, mode=0o700)
    _owned(directory)
    return directory / key


def _read(entry: Path, key: str) -> dict | None:
    try:
        _owned(entry)
        record_path = entry / "record.json"
        if record_path.is_symlink():
            return None
        record = json.loads(record_path.read_text())
        if (record["key"] == key and fingerprint(record["inputs"]) == key
                and isinstance(record["origin"], dict) and record["files"] == files_in(entry / "payload")):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def prepare(request: Path, name: str, inputs: dict, observations: dict, origin: dict,
            output: Path, *, clean: bool = False, enabled: bool = True) -> bool:
    """Restore an exact component hit into a fresh output and retain its real provenance.

    Args:
        request: Build-local receipt consumed only after the full build commits.
        name: Stable component family.
        inputs: Content and ABI identity with logical source paths.
        observations: Actual input paths and hashes for publication-time revalidation.
        origin: Current source and environment identity, separate from the key.
        output: Empty, build-owned output directory.
        clean: Bypass existing entries without removing shared cache contents.
        enabled: Whether the input environment is supported for reuse.
    """
    started = time.monotonic()
    key = fingerprint(inputs)
    record = None
    root = cache_root() if enabled else None
    if enabled and not clean:
        entry = _entry(root, name, key)
        with _lock(entry.parent / (key + ".lock")):
            record = _read(entry, key)
            if record is not None:
                if any(path.is_symlink() for path in (output, *output.parents)):
                    raise ValueError(f"Redirected component restore output: {output}")
                if output.exists() and any(output.iterdir()):
                    raise ValueError(f"Component restore requires an empty output: {output}")
                shutil.copytree(entry / "payload", output, dirs_exist_ok=True)
                if files_in(output) != record["files"]:
                    raise ValueError("Component changed during restore")
    receipt = {"schema": 1, "name": name, "key": key, "inputs": inputs,
               "observations": observations, "origin": record["origin"] if record else origin,
               "current_origin": origin, "reused": record is not None, "enabled": enabled,
               "output": str(output), "cache_root": str(root) if root else None,
               "files": record["files"] if record else {}, "lookup_seconds": time.monotonic() - started,
               "prepared_at": time.time(), "compile_seconds": 0.0}
    atomic_json(request, receipt)
    state = "hit" if record else "miss" if enabled else "miss; reuse disabled for incomplete or unsupported inputs"
    message(f"Component {name}: {state} ({key[:12]})")
    return record is not None


def stage(request: Path) -> None:
    """Retain a successful component privately; failed overall builds cannot publish it."""
    receipt = json.loads(request.read_text())
    if receipt["reused"]:
        return
    output = Path(receipt["output"])
    receipt["files"] = files_in(output)
    receipt["compile_seconds"] = time.time() - receipt["prepared_at"]
    if not receipt["enabled"]:
        atomic_json(request, receipt)
        return
    staged = request.with_suffix(".payload")
    if staged.exists():
        raise FileExistsError(staged)
    shutil.copytree(output, staged)
    if files_in(staged) != receipt["files"]:
        raise ValueError("Component changed during staging")
    receipt["staged"] = str(staged)
    atomic_json(request, receipt)


def provenance(receipt: dict) -> dict:
    """Expose component truth without copying large input manifests into build_info."""
    result = {key: receipt[key] for key in
              ("name", "key", "origin", "reused", "files", "lookup_seconds", "compile_seconds")}
    result["input_scan_seconds"] = receipt.get("input_scan_seconds", 0.0)
    return result


def publish(requests: list[Path]) -> None:
    """Revalidate observed inputs once and atomically publish only committed build components."""
    receipts = [json.loads(path.read_text()) for path in requests]
    observed = {}
    for receipt in receipts:
        if not receipt["enabled"]:
            continue
        for path, expected in receipt["observations"].items():
            if path in observed and observed[path] != expected:
                raise ValueError(f"Inconsistent component input observations: {path}")
            observed[path] = expected
    for name, expected in observed.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Component input changed; refusing publication: {name}")
    for receipt in receipts:
        if receipt["enabled"] and not receipt["reused"] and "staged" in receipt:
            _publish_one(receipt)


def _publish_one(receipt: dict) -> None:
    root = Path(receipt["cache_root"])
    _owned(root)
    entry = _entry(root, receipt["name"], receipt["key"])
    with _lock(entry.parent / (entry.name + ".lock")):
        if _read(entry, receipt["key"]) is not None:
            return
        staged = Path(receipt["staged"])
        if files_in(staged) != receipt["files"]:
            raise ValueError("Staged component changed before publication")
        temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=entry.parent))
        try:
            shutil.copytree(staged, temporary / "payload")
            atomic_json(temporary / "record.json", {key: receipt[key] for key in
                        ("schema", "name", "key", "inputs", "origin", "files")})
            if _read(temporary, receipt["key"]) is None:
                raise ValueError("New component entry failed verification")
            if entry.exists() or entry.is_symlink():
                if entry.is_symlink():
                    raise ValueError("Refusing redirected component entry")
                _owned(entry)
                shutil.rmtree(entry)
            os.replace(temporary, entry)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
