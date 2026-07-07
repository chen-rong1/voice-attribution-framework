"""Small config helpers used by the first framework iteration."""

from __future__ import annotations

from pathlib import Path


def load_simple_yaml_map(config_path: Path) -> dict[str, str]:
    """Load a flat `key: value` YAML-like config without extra dependencies."""

    values: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        if not key or not _:
            continue
        values[key.strip()] = value.strip()
    return values
