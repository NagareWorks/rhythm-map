#!/usr/bin/env python3
"""Verify every file in a Rhythm Map distribution directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(package: Path) -> dict[str, object]:
    package = package.resolve(strict=True)
    sums_path = package / "SHA256SUMS"
    expected: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        path = PurePosixPath(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute()
            or ".." in path.parts
            or relative in expected
        ):
            raise ValueError(f"invalid SHA256SUMS entry on line {line_number}")
        expected[relative] = digest

    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unlisted = sorted(actual - set(expected))
        raise ValueError(f"package file set mismatch; missing={missing}, unlisted={unlisted}")

    for relative, digest in expected.items():
        if sha256(package / Path(*PurePosixPath(relative).parts)) != digest:
            raise ValueError(f"SHA-256 mismatch: {relative}")

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported distribution manifest schema")
    commit = manifest.get("git_commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("distribution manifest does not contain a full Git SHA")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="unpacked distribution directory")
    args = parser.parse_args()
    manifest = verify(args.package)
    print(
        f"verified {manifest['package']} {manifest['version']} "
        f"for {manifest['target']} at {manifest['git_commit']}"
    )


if __name__ == "__main__":
    main()
