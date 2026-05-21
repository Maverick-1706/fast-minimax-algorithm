#export.py
"""Export utilities for benchmark results.

Every run produces machine-readable output with full metadata (seed, dims, ε,
hardware, JAX version, commit hash).
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import jax
from minimax_aipe.problem import BenchmarkProblem
from benchmarks.memory import MemoryResult
from benchmarks.results import BenchmarkResult as _BenchmarkResult


# ── Metadata ──────────────────────────────────────────────────────────────


def _git_info() -> dict[str, str]:
    """Capture git commit hash and dirty flag."""
    info: dict[str, str] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["short_commit"] = info["commit"][:8]
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["dirty"] = bool(dirty)
    except Exception:
        info["commit"] = "unknown"
        info["short_commit"] = "unknown"
        info["dirty"] = "unknown"
    return info


def collect_metadata(
    *,
    epsilon: float,
    n_repeats: int,
    seed: int | None,
    section: str,
    dims: str | None,
    names: str | None,
    quick: bool,
    problems: list[BenchmarkProblem],
) -> dict:
    """Collect environment and run metadata.

    Returns a dict suitable for JSON serialisation.
    """
    devices = []
    for d in jax.local_devices():
        dev: dict = {
            "device_kind": getattr(d, "device_kind", "unknown"),
            "platform": d.platform,
            "id": d.id,
        }
        try:
            dev["memory_limit"] = d.memory_limit()
        except Exception:
            pass
        devices.append(dev)

    git = _git_info()

    return {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "jax_version": jax.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "devices": devices,
        "jax_platforms": os.environ.get("JAX_PLATFORMS", "(default)"),
        "git": git,
        "run": {
            "epsilon": epsilon,
            "repeats": n_repeats,
            "seed": seed,
            "quick": quick,
            "section": section,
            "dims": dims,
            "names": names,
        },
        "problems": [{"name": p.name, "dim": p.dim} for p in problems],
    }


def write_metadata(meta: dict, path: str = "metadata.json") -> None:
    """Write metadata JSON to *path*."""
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ── Flattening ────────────────────────────────────────────────────────────


def flatten_speed_rows(rows: list[_BenchmarkResult]) -> list[dict]:
    """Flatten BenchmarkResult list into flat dicts for CSV/JSON export."""
    return [r.to_dict() for r in rows]


def flatten_jit_rows(rows: list[dict]) -> list[dict]:
    flat = []
    for r in rows:
        jit = r["jit"]
        eager = r["eager"]
        jit_lo, jit_hi = jit.get("ci", (0.0, 0.0))
        eager_lo, eager_hi = eager.get("ci", (0.0, 0.0))
        flat.append({
            "name": r["name"], "dim": r["dim"],
            "jit_mean": jit["mean"], "jit_std": jit.get("std", 0.0),
            "jit_ci_lo": jit_lo, "jit_ci_hi": jit_hi,
            "eager_mean": eager["mean"], "eager_std": eager.get("std", 0.0),
            "eager_ci_lo": eager_lo, "eager_ci_hi": eager_hi,
            "speedup": r["speedup"],
        })
    return flat


def flatten_memory_rows(rows: list[MemoryResult]) -> list[dict]:
    """Flatten MemoryResult list into flat dicts, preserving hardware tracking metrics."""
    flat = []
    for r in rows:
        flat.append({
            "name": r.name,
            "dim": r.dim,
            "solver": r.solver,
            "peak_mb": r.peak_bytes / (1024 * 1024),
            "jax_mb": r.jax_bytes / (1024 * 1024),
            "device_peak_mb": r.device_bytes_peak / (1024 * 1024),
            "device_delta_mb": r.device_bytes_delta / (1024 * 1024),
            "utilization": r.device_utilization,
        })
    return flat


def flatten_convergence_rows(rows: list[_BenchmarkResult]) -> list[dict]:
    """Flatten BenchmarkResult list into flat dicts for CSV/JSON export."""
    return [r.to_dict() for r in rows]


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_scaling_rows(rows: list[_BenchmarkResult]) -> list[dict]:
    """Flatten BenchmarkResult list into flat dicts for CSV/JSON export."""
    return [r.to_dict() for r in rows]


def flatten_ablation_rows(rows: list[_BenchmarkResult]) -> list[dict]:
    """Flatten BenchmarkResult list into flat dicts for CSV/JSON export."""
    return [r.to_dict() for r in rows]


# ── Writers ───────────────────────────────────────────────────────────────


def _default_json(obj):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj.tolist()
    return str(obj)


def write_json(data: dict, path: str | None) -> None:
    """Write JSON to file or stdout."""
    payload = json.dumps(data, indent=2, default=_default_json)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(payload)
        print(f"  JSON written to {path}")
    else:
        print(payload)


def write_csv(data: dict[str, list[dict]], path: str | None) -> None:
    """Write CSV to file or stdout.  Each section becomes a '# section_name' header."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO() if path is None else open(path, "w", newline="")
    try:
        for section_name, rows in data.items():
            if not rows or section_name == "metadata":
                continue
            if isinstance(rows, dict):
                rows = [rows]
            buf.write(f"# {section_name}\n")
            if rows:
                writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            buf.write("\n")

        if path:
            print(f"  CSV written to {path}")
        else:
            print(buf.getvalue())
    finally:
        buf.close()


def export_results(data: dict, fmt: str, path: str | None) -> None:
    """Dispatch to the appropriate writer."""
    if fmt == "json":
        write_json(data, path)
    elif fmt == "csv":
        write_csv(data, path)
    else:
        raise ValueError(f"Unknown export format {fmt!r}; expected 'csv' or 'json'.")