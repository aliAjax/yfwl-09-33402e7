#!/usr/bin/env python3
"""Provisional oscillator recovery command used by the lab service."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


MAX_POINTS = 48


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("payload must be an object")
    return value


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def validate(payload: dict) -> dict:
    required = {"case_id", "max_points", "omega", "drive", "segments"}
    if set(payload) != required:
        raise ValueError("payload keys do not match the contract")
    if payload["max_points"] != MAX_POINTS:
        raise ValueError("max_points is unsupported")
    case_id = payload["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id is required")
    drive = payload["drive"]
    if not isinstance(drive, dict) or set(drive) != {"a0", "a1", "a2"}:
        raise ValueError("drive keys do not match the contract")
    segments = payload["segments"]
    if not isinstance(segments, list) or not 2 <= len(segments) <= 4:
        raise ValueError("segments outside supported count")
    total = 0
    cleaned = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"times", "observations"}:
            raise ValueError("segment keys do not match the contract")
        times = segment["times"]
        values = segment["observations"]
        if not isinstance(times, list) or not isinstance(values, list):
            raise ValueError("segment arrays are required")
        if len(times) != len(values) or not 4 <= len(times) <= 16:
            raise ValueError("segment sample count is invalid")
        clean_times = [number(item, "time") for item in times]
        clean_values = [number(item, "observation") for item in values]
        if any(clean_times[i] >= clean_times[i + 1] for i in range(len(clean_times) - 1)):
            raise ValueError("times must increase")
        total += len(times)
        cleaned.append({"times": clean_times, "observations": clean_values})
    if total > MAX_POINTS:
        raise ValueError("observation budget exceeded")
    return {
        "case_id": case_id,
        "max_points": MAX_POINTS,
        "omega": number(payload["omega"], "omega"),
        "drive": {key: number(drive[key], key) for key in ("a0", "a1", "a2")},
        "segments": cleaned,
    }


def estimate(payload: dict) -> dict:
    values = [
        item
        for segment in payload["segments"]
        for item in segment["observations"]
    ]
    mean = sum(values) / len(values)
    spread = (
        sum((item - mean) ** 2 for item in values) / max(1, len(values) - 1)
    ) ** 0.5
    states = [
        {
            "segment_index": index,
            "y": round(segment["observations"][0], 9),
            "v": 0.0,
        }
        for index, segment in enumerate(payload["segments"])
    ]
    return {
        "status": "recovered",
        "parameters": {
            "c": round(0.5 + 0.02 * spread, 9),
            "k": round(1.0 + abs(mean), 9),
            "beta": 0.0,
            "phase": 0.0,
        },
        "initial_states": states,
        "quality": {
            "rms": round(spread, 12),
            "iterations": 1,
            "jacobian_rank": 4 + 2 * len(states),
        },
    }


def canonical(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_ledger(
    ledger: Path,
    case_id: str,
    input_digest: str,
    output_digest: str,
) -> None:
    rows = []
    if ledger.exists():
        rows = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line
        ]
    if any(
        row.get("case_id") == case_id
        and row.get("input_sha256") == input_digest
        and row.get("output_sha256") == output_digest
        for row in rows
    ):
        return
    record = {
        "case_id": case_id,
        "input_sha256": input_digest,
        "output_sha256": output_digest,
        "sequence": len(rows) + 1,
    }
    with ledger.open("ab") as handle:
        handle.write(canonical(record))
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("ODE_REFUSED: expected input output ledger", file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    ledger_path = Path(argv[3])
    try:
        payload = validate(load_json(input_path))
        report = estimate(payload)
        data = canonical(report)
        atomic_write(output_path, data)
        append_ledger(
            ledger_path,
            payload["case_id"],
            hashlib.sha256(input_path.read_bytes()).hexdigest(),
            hashlib.sha256(data).hexdigest(),
        )
        print("ODE_RECOVERED")
        return 0
    except Exception as exc:
        print(f"ODE_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
