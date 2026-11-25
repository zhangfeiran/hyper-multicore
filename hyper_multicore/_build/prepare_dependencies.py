# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Verify and extract the locked native build sources shipped with the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO


_LOCK_PATH = Path(__file__).with_name("dependencies.lock.json")
_ARCHIVE_DIR = _LOCK_PATH.with_name("dependency_sources")


def _parse_args(dependencies: dict[str, Any]) -> argparse.Namespace:
    """Select the archives to verify or one archive to extract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dependency",
        action="append",
        required=True,
        choices=tuple(dependencies),
        help="Dependency to prepare. Repeat the option to prepare several dependencies.",
    )
    parser.add_argument("--source-dir", type=Path, help="Verify a local upstream Git repository")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true", help="Report verified source identities (default)")
    action.add_argument("--export-dir", type=Path, help="Extract one locked archive into a new directory")
    return parser.parse_args()


def main() -> int:
    """Verify or export the requested pinned source inputs and report their identities."""
    dependencies = load_dependency_locks()
    args = _parse_args(dependencies)
    if args.source_dir and len(args.dependency) != 1:
        raise ValueError("--source-dir can only be used with one --dependency value.")
    if args.export_dir and (args.source_dir or len(args.dependency) != 1
                            or "revisions" in dependencies[args.dependency[0]]):
        raise ValueError("--export-dir requires one single-revision dependency and uses its shipped archive.")
    prepared: dict[str, Any] = {}
    for name in dict.fromkeys(args.dependency):
        dependency = dependencies[name]
        verified = {}
        for source_name, revision in named_revisions(name, dependency).items():
            if args.source_dir:
                verified[source_name] = verify_git_dependency(revision, args.source_dir, source_name)
                continue
            archive = verify_source_archive(source_name, revision, export_dir=args.export_dir)
            verified[source_name] = {
                "dependency": source_name, "version": revision["version"], "repository": revision["repository"],
                "commit": revision["commit"], "tree": revision["tree"],
                "archive": str(archive), "archive_sha256": revision["archive_sha256"], "patched": False,
            }
        if "revisions" not in dependency:
            prepared[name] = verified[name]
            continue
        record = {"repository": dependency["repository"],
                  "revisions": {key.split(".", 1)[1]: value for key, value in verified.items()}}
        if args.source_dir:
            record["source_dir"] = str(args.source_dir.resolve())
        prepared[name] = record
    print(json.dumps(prepared, sort_keys=True))
    return 0


def verify_git_dependency(
    dependency_lock: dict[str, Any],
    source_dir: str | Path,
    dependency_name: str | None = None,
) -> dict[str, Any]:
    """Verify a locked commit independently of the shared worktree's current HEAD.

    Args:
        dependency_lock: Dependency entry from ``dependencies.lock.json``.
        source_dir: Existing upstream Git repository to verify.
        dependency_name: Stable lock name; inferred from the repository when omitted.

    Returns:
        Verified source identity suitable for a run record.

    Raises:
        ValueError: If the source is missing, modified, or does not match the lock.
    """
    source_dir = Path(source_dir).resolve()
    worktree = (source_dir / ".git").exists()
    if not worktree and not ((source_dir / "HEAD").is_file() and (source_dir / "objects").is_dir()):
        raise ValueError(f"Dependency source is not a Git repository: {source_dir}")
    commit = dependency_lock["commit"]
    actual_commit = _git_output(source_dir, "rev-parse", f"{commit}^{{commit}}")
    actual_tree = _git_output(source_dir, "rev-parse", f"{commit}^{{tree}}")
    if actual_commit != dependency_lock["commit"]:
        raise ValueError(f"Dependency commit mismatch: expected={dependency_lock['commit']}, actual={actual_commit}")
    if actual_tree != dependency_lock["tree"]:
        raise ValueError(f"Dependency tree mismatch: expected={dependency_lock['tree']}, actual={actual_tree}")
    if worktree and _git_output(source_dir, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Dependency source contains tracked modifications; upstream patching is not allowed.")
    repository_name = Path(dependency_lock["repository"]).stem
    archive_hash = _git_archive_sha256(
        source_dir,
        dependency_lock["commit"],
        f"{repository_name}-{dependency_lock['version']}/",
    )
    expected_archive_hashes = {
        dependency_lock["git_archive_tar_sha256"],
        *dependency_lock.get("git_archive_tar_sha256_compat", []),
    }
    if archive_hash not in expected_archive_hashes:
        raise ValueError(
            "Dependency archive hash mismatch: "
            f"expected one of {sorted(expected_archive_hashes)}, actual={archive_hash}"
        )
    return {
        "dependency": dependency_name or repository_name.replace("-", "_"),
        "version": dependency_lock["version"],
        "repository": dependency_lock["repository"],
        "source_dir": str(source_dir),
        "commit": actual_commit,
        "tree": actual_tree,
        "git_archive_tar_sha256": archive_hash,
        "patched": False,
    }


def load_dependency_locks(lock_path: Path | None = None) -> dict[str, Any]:
    """Read schema 2 and validate archive identities and repository declarations."""
    lock = json.loads((lock_path or _LOCK_PATH).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 2:
        raise ValueError("Native dependency lock must use schema_version 2")
    dependencies = lock["components"]["multicore"]
    repositories = {}
    for name, dependency in dependencies.items():
        repository = dependency["repository"]
        if repository in repositories:
            raise ValueError(
                f"Repository {repository} is declared by both {repositories[repository]} and {name}; "
                "declare additional versions under revisions instead."
            )
        repositories[repository] = name
        for source_name, revision in named_revisions(name, dependency).items():
            digest = revision.get("archive_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Invalid archive_sha256 for {source_name}")
    return dependencies


def dependency_revisions(dependency_lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve single-version or named revisions into complete verification entries."""
    if "revisions" not in dependency_lock:
        return {"default": dependency_lock}
    revisions = dependency_lock["revisions"]
    if not isinstance(revisions, dict) or not revisions:
        raise ValueError("Dependency revisions must be a nonempty mapping")
    if any(not isinstance(revision, dict) or "repository" in revision for revision in revisions.values()):
        raise ValueError("Each revision must be an object; repository belongs to the parent entry")
    return {name: {**revision, "repository": dependency_lock["repository"]}
            for name, revision in revisions.items()}


def named_revisions(name: str, dependency: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return revision locks keyed by their archive names."""
    if "revisions" not in dependency:
        return {name: dependency}
    return {f"{name}.{label}": revision for label, revision in dependency_revisions(dependency).items()}


def verify_source_archive(name: str, revision: dict[str, Any], *, export_dir: Path | None = None) -> Path:
    """Validate the shipped archive and optionally extract it into a new build directory."""
    path = _ARCHIVE_DIR / f"{name}.tar.gz"
    with path.open("rb") as source:
        actual = _sha256(source)
        expected = revision["archive_sha256"]
        if actual != expected:
            raise ValueError(f"Source archive SHA256 mismatch for {path}: expected={expected}, actual={actual}")
        source.seek(0)
        try:
            with tarfile.open(fileobj=source, mode="r:gz") as archive:
                if archive.pax_headers.get("comment") != revision["commit"]:
                    raise ValueError(f"Source archive does not identify locked commit {revision['commit']}: {path}")
                if export_dir is not None:
                    _export_archive(archive, export_dir)
        except tarfile.TarError as error:
            raise ValueError(f"Cannot read source archive {path}: {error}") from error
    return path


def _export_archive(archive: tarfile.TarFile, destination: Path) -> None:
    """Publish a complete source copy only after checking the archive paths and types."""
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Source export directory already exists: {destination}")
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsafe source archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "source"
        archive.extractall(staged, members)
        staged.rename(destination)


def _git_output(source_dir: Path, *args: str) -> str:
    """Run one read-only Git inspection and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=source_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Git inspection failed ({' '.join(args)}) in {source_dir}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_archive_sha256(source_dir: Path, commit: str, archive_prefix: str) -> str:
    """Hash a deterministic uncompressed Git archive without creating a tarball."""
    with subprocess.Popen(
        ["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
         "archive", "--format=tar", f"--prefix={archive_prefix}", commit],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Cannot read git archive output.")
        digest = _sha256(process.stdout)
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise ValueError(f"Git archive failed: {stderr.decode(errors='replace').strip()}")
    return digest


def _sha256(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"[HP-NATIVE-DEPENDENCY-ERROR] {error}\n")
        raise SystemExit(1) from error
