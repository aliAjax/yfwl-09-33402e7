#!/usr/bin/env python3
"""Recover non-autonomous oscillator parameters and hidden initial states."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


ENTRY = "/app/ode_recover.py"
MAX_POINTS = 48
MIN_SEGMENTS = 2
MAX_SEGMENTS = 4
MIN_SAMPLES = 4
MAX_SAMPLES = 16
TOP_KEYS = {"case_id", "max_points", "omega", "drive", "segments"}
DRIVE_KEYS = {"a0", "a1", "a2"}
SEGMENT_KEYS = {"times", "observations"}


class RecoveryError(ValueError):
    """Raised when the input cannot define a recoverable experiment."""


def _reject_constant(token: str) -> None:
    raise RecoveryError(f"non-finite JSON number {token}")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError(f"duplicate key {key}")
        result[key] = value
    return result


def load_payload(path: Path) -> dict:
    """Load strict UTF-8 JSON and reject duplicate keys."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecoveryError("input cannot be read") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RecoveryError("input is not valid JSON") from exc
    return require_mapping(payload, "top-level payload")


def require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be an object")
    return value


def require_exact_keys(value: dict, expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing:
        raise RecoveryError(f"{label} missing keys: {','.join(missing)}")
    if extra:
        raise RecoveryError(f"{label} extra keys: {','.join(extra)}")


def require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RecoveryError(f"{label} must be finite")
    return number


def require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryError(f"{label} must be an integer")
    return value


def parse_input(payload: dict) -> dict:
    """Validate the experiment contract before touching persistent state."""
    require_exact_keys(payload, TOP_KEYS, "payload")
    case_id = payload["case_id"]
    if not isinstance(case_id, str) or not case_id or len(case_id) > 48:
        raise RecoveryError("case_id must be a non-empty string")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in case_id):
        raise RecoveryError("case_id has unsupported characters")
    max_points = require_integer(payload["max_points"], "max_points")
    if max_points != MAX_POINTS:
        raise RecoveryError(f"max_points must equal {MAX_POINTS}")
    omega = require_number(payload["omega"], "omega")
    if not 0.2 <= omega <= 4.0:
        raise RecoveryError("omega outside supported range")
    drive = require_mapping(payload["drive"], "drive")
    require_exact_keys(drive, DRIVE_KEYS, "drive")
    a0 = require_number(drive["a0"], "drive.a0")
    a1 = require_number(drive["a1"], "drive.a1")
    a2 = require_number(drive["a2"], "drive.a2")
    if not -1.0 <= a0 <= 1.0 or not -1.5 <= a1 <= 1.5 or not -1.0 <= a2 <= 1.0:
        raise RecoveryError("drive amplitudes outside supported range")
    raw_segments = payload["segments"]
    if not isinstance(raw_segments, list):
        raise RecoveryError("segments must be an array")
    if not MIN_SEGMENTS <= len(raw_segments) <= MAX_SEGMENTS:
        raise RecoveryError("segments outside supported count")
    segments: list[dict] = []
    segment_signatures: set[str] = set()
    total_points = 0
    for index, raw_segment in enumerate(raw_segments):
        segment = require_mapping(raw_segment, f"segments[{index}]")
        require_exact_keys(segment, SEGMENT_KEYS, f"segments[{index}]")
        raw_times = segment["times"]
        raw_values = segment["observations"]
        if not isinstance(raw_times, list) or not isinstance(raw_values, list):
            raise RecoveryError(f"segments[{index}] arrays are required")
        if len(raw_times) != len(raw_values):
            raise RecoveryError(f"segments[{index}] lengths differ")
        if not MIN_SAMPLES <= len(raw_times) <= MAX_SAMPLES:
            raise RecoveryError(f"segments[{index}] sample count outside range")
        times = [require_number(value, f"segments[{index}].times") for value in raw_times]
        observations = [
            require_number(value, f"segments[{index}].observations")
            for value in raw_values
        ]
        if abs(times[0]) > 1e-12:
            raise RecoveryError(f"segments[{index}] must begin at zero")
        if any(times[j] >= times[j + 1] for j in range(len(times) - 1)):
            raise RecoveryError(f"segments[{index}] times must increase")
        if any(not -3.0 <= value <= 3.0 for value in observations):
            raise RecoveryError(f"segments[{index}] observations outside range")
        signature = json.dumps(
            {"times": times, "observations": observations},
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in segment_signatures:
            raise RecoveryError("segments must be independently identifiable")
        segment_signatures.add(signature)
        total_points += len(times)
        segments.append({"times": times, "observations": observations})
    if total_points > max_points:
        raise RecoveryError("observation budget exceeded")
    return {
        "case_id": case_id,
        "max_points": max_points,
        "omega": omega,
        "drive": {"a0": a0, "a1": a1, "a2": a2},
        "segments": segments,
        "total_points": total_points,
    }


def drive_value(t: float, phase: float, experiment: dict) -> float:
    drive = experiment["drive"]
    omega = experiment["omega"]
    return (
        drive["a0"]
        + drive["a1"] * math.sin(omega * t + phase)
        + drive["a2"] * math.cos(2.0 * omega * t)
    )


def derivatives(
    t: float,
    y: float,
    velocity: float,
    parameters: list[float],
    experiment: dict,
) -> tuple[float, float]:
    damping, stiffness, cubic, phase = parameters
    acceleration = (
        drive_value(t, phase, experiment)
        - damping * velocity
        - stiffness * y
        - cubic * y * y * y
    )
    return velocity, acceleration


def advance(
    times: list[float],
    initial_y: float,
    initial_v: float,
    parameters: list[float],
    experiment: dict,
) -> list[float]:
    """Integrate with fixed fine RK4 substeps for reproducible Jacobians."""
    values = [initial_y]
    t = times[0]
    y = initial_y
    velocity = initial_v
    for target in times[1:]:
        interval = target - t
        steps = max(1, int(math.ceil(interval / 0.025)))
        step = interval / steps
        for _ in range(steps):
            k1y, k1v = derivatives(t, y, velocity, parameters, experiment)
            k2y, k2v = derivatives(
                t + step / 2,
                y + step * k1y / 2,
                velocity + step * k1v / 2,
                parameters,
                experiment,
            )
            k3y, k3v = derivatives(
                t + step / 2,
                y + step * k2y / 2,
                velocity + step * k2v / 2,
                parameters,
                experiment,
            )
            k4y, k4v = derivatives(
                t + step,
                y + step * k3y,
                velocity + step * k3v,
                parameters,
                experiment,
            )
            y += step * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
            velocity += step * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            t += step
        values.append(y)
    return values


def residual_vector(
    theta: list[float],
    experiment: dict,
) -> list[float]:
    parameters = theta[:4]
    residuals: list[float] = []
    for index, segment in enumerate(experiment["segments"]):
        initial_y = theta[4 + 2 * index]
        initial_v = theta[5 + 2 * index]
        predicted = advance(
            segment["times"],
            initial_y,
            initial_v,
            parameters,
            experiment,
        )
        residuals.extend(
            predicted_value - observed_value
            for predicted_value, observed_value in zip(
                predicted, segment["observations"]
            )
        )
    return residuals


def sum_squares(values: list[float]) -> float:
    return sum(value * value for value in values)


def normal_equations(
    theta: list[float],
    residual: list[float],
    experiment: dict,
) -> tuple[list[list[float]], list[float]]:
    dimension = len(theta)
    rows = len(residual)
    jacobian = [[0.0] * dimension for _ in range(rows)]
    for column in range(dimension):
        epsilon = 1e-5 * max(1.0, abs(theta[column]))
        shifted = theta[:]
        shifted[column] += epsilon
        shifted_residual = residual_vector(shifted, experiment)
        for row in range(rows):
            jacobian[row][column] = (
                shifted_residual[row] - residual[row]
            ) / epsilon
    normal = [
        [
            sum(jacobian[row][left] * jacobian[row][right] for row in range(rows))
            for right in range(dimension)
        ]
        for left in range(dimension)
    ]
    gradient = [
        sum(jacobian[row][column] * residual[row] for row in range(rows))
        for column in range(dimension)
    ]
    return normal, gradient


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    size = len(rhs)
    augmented = [matrix[row][:] + [rhs[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-13:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def matrix_rank(matrix: list[list[float]], tolerance: float = 1e-9) -> int:
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = max(range(rank, rows), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        divisor = work[rank][column]
        work[rank] = [value / divisor for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            work[row] = [
                work[row][index] - factor * work[rank][index]
                for index in range(columns)
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def parameter_bounds_hold(theta: list[float]) -> bool:
    damping, stiffness, cubic, phase = theta[:4]
    if not 0.05 <= damping <= 2.5:
        return False
    if not 0.8 <= stiffness <= 4.0:
        return False
    if not -0.5 <= cubic <= 0.8:
        return False
    if not -math.pi <= phase <= math.pi:
        return False
    for index in range(4, len(theta), 2):
        if not -2.0 <= theta[index] <= 2.0:
            return False
        if not -2.0 <= theta[index + 1] <= 2.0:
            return False
    return True


def recover(experiment: dict) -> dict:
    """Run damped Gauss-Newton with a deterministic multi-start policy."""
    segment_count = len(experiment["segments"])
    dimension = 4 + 2 * segment_count
    seeds: list[list[float]] = []
    base_parameter_sets = ([0.8, 1.2, 0.1, 0.0], [0.2, 2.2, 0.4, 0.7])
    for base in base_parameter_sets:
        seed = list(base)
        for segment in experiment["segments"]:
            seed.extend([segment["observations"][0], 0.0])
        seeds.append(seed)
    best_theta: list[float] | None = None
    best_cost = math.inf
    best_iterations = 0
    best_normal: list[list[float]] | None = None
    for seed in seeds:
        theta = seed[:]
        damping = 1e-3
        iterations = 0
        for iterations in range(1, 71):
            residuals = residual_vector(theta, experiment)
            cost = sum_squares(residuals)
            normal, gradient = normal_equations(theta, residuals, experiment)
            regularized = [row[:] for row in normal]
            for index in range(dimension):
                regularized[index][index] += damping * max(
                    1.0, regularized[index][index]
                )
            delta = solve_linear(regularized, [-value for value in gradient])
            if delta is None:
                damping = min(1e9, damping * 8.0)
                continue
            candidate = [theta[index] + delta[index] for index in range(dimension)]
            if not parameter_bounds_hold(candidate):
                damping = min(1e9, damping * 5.0)
                continue
            candidate_residuals = residual_vector(candidate, experiment)
            candidate_cost = sum_squares(candidate_residuals)
            if candidate_cost < cost:
                theta = candidate
                damping = max(1e-10, damping * 0.35)
                if max(abs(value) for value in delta) < 1e-9:
                    break
            else:
                damping = min(1e9, damping * 5.0)
                if damping >= 1e8:
                    break
        final_residuals = residual_vector(theta, experiment)
        final_cost = sum_squares(final_residuals)
        final_normal, _ = normal_equations(theta, final_residuals, experiment)
        if final_cost < best_cost:
            best_theta = theta
            best_cost = final_cost
            best_iterations = iterations
            best_normal = final_normal
    if best_theta is None or best_normal is None:
        raise RecoveryError("nonlinear solver did not produce a candidate")
    rank = matrix_rank(best_normal)
    rms = math.sqrt(best_cost / experiment["total_points"])
    if rank < dimension:
        raise RecoveryError("observation design is rank deficient")
    if rms > 2e-3:
        raise RecoveryError("observations do not close the recovery residual")
    parameters = {
        "c": round(best_theta[0], 9),
        "k": round(best_theta[1], 9),
        "beta": round(best_theta[2], 9),
        "phase": round(best_theta[3], 9),
    }
    initial_states = [
        {
            "segment_index": index,
            "y": round(best_theta[4 + 2 * index], 9),
            "v": round(best_theta[5 + 2 * index], 9),
        }
        for index in range(segment_count)
    ]
    return {
        "status": "recovered",
        "parameters": parameters,
        "initial_states": initial_states,
        "quality": {
            "rms": round(rms, 12),
            "iterations": best_iterations,
            "jacobian_rank": rank,
        },
    }


def canonical_json(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    """Write through a sibling temporary file and atomically replace the target."""
    parent = path.parent
    if not parent.is_dir():
        raise RecoveryError("output parent does not exist")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        # Recover a checkpoint after an interrupted append by dropping the
        # incomplete tail; complete records remain usable.
        complete = raw[: raw.rfind(b"\n") + 1]
        temporary = path.with_name(path.name + ".recover")
        atomic_write(temporary, complete)
        os.replace(temporary, path)
        raw = complete
    entries: list[dict] = []
    for line in raw.splitlines():
        if not line:
            raise RecoveryError("ledger contains an empty record")
        try:
            item = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryError("ledger is not valid JSONL") from exc
        if not isinstance(item, dict):
            raise RecoveryError("ledger record must be an object")
        require_exact_keys(
            item,
            {"case_id", "input_sha256", "output_sha256", "sequence"},
            "ledger record",
        )
        entries.append(item)
    return entries


def append_ledger(
    path: Path,
    case_id: str,
    input_digest: str,
    output_digest: str,
) -> bool:
    """Append once under an exclusive lock, preserving failed-run history."""
    parent = path.parent
    if not parent.is_dir():
        raise RecoveryError("ledger parent does not exist")
    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        entries = load_ledger(path)
        duplicate = any(
            entry["case_id"] == case_id
            and entry["input_sha256"] == input_digest
            and entry["output_sha256"] == output_digest
            for entry in entries
        )
        if duplicate:
            return False
        sequence = max((int(entry["sequence"]) for entry in entries), default=0) + 1
        record = {
            "case_id": case_id,
            "input_sha256": input_digest,
            "output_sha256": output_digest,
            "sequence": sequence,
        }
        with path.open("ab", buffering=0) as handle:
            handle.write(canonical_json(record))
            os.fsync(handle.fileno())
        return True


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finish_success(
    input_path: Path,
    output_path: Path,
    ledger_path: Path,
    payload: dict,
) -> None:
    report = recover(payload)
    output_bytes = canonical_json(report)
    input_bytes = input_path.read_bytes()
    ledger_record = {
        "case_id": payload["case_id"],
        "input_sha256": digest_bytes(input_bytes),
        "output_sha256": digest_bytes(output_bytes),
        "sequence": 0,
    }
    atomic_write(output_path, output_bytes)
    try:
        append_ledger(
            ledger_path,
            ledger_record["case_id"],
            ledger_record["input_sha256"],
            ledger_record["output_sha256"],
        )
    except Exception:
        # The output is only transiently newer than the ledger. On any ledger
        # failure, restore the previous output bytes to preserve rollback.
        raise
    print("ODE_RECOVERED")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("ODE_REFUSED: expected input output ledger", file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    ledger_path = Path(argv[3])
    output_before = output_path.read_bytes() if output_path.is_file() else None
    ledger_before = ledger_path.read_bytes() if ledger_path.is_file() else None
    try:
        payload = parse_input(load_payload(input_path))
        finish_success(input_path, output_path, ledger_path, payload)
        return 0
    except Exception as exc:
        # Failure rollback restores both externally visible artifacts.
        if output_before is None:
            try:
                output_path.unlink()
            except OSError:
                pass
        else:
            try:
                atomic_write(output_path, output_before)
            except Exception:
                pass
        if ledger_before is None:
            try:
                ledger_path.unlink()
            except OSError:
                pass
        else:
            try:
                atomic_write(ledger_path, ledger_before)
            except Exception:
                pass
        reason = str(exc) or exc.__class__.__name__
        print(f"ODE_REFUSED: {reason}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
