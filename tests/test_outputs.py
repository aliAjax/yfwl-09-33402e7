"""Behavior checks for the non-autonomous oscillator recovery task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


CANDIDATE = os.environ.get("ODE_ENTRY", "/app/ode_recover.py")
MAX_POINTS = 48


def drive(t: float, phase: float, packet: dict) -> float:
    omega = packet["omega"]
    params = packet["drive"]
    return (
        params["a0"]
        + params["a1"] * math.sin(omega * t + phase)
        + params["a2"] * math.cos(2.0 * omega * t)
    )


def integrate(times: list[float], y0: float, v0: float, truth: dict) -> list[float]:
    damping, stiffness, cubic, phase = truth["parameters"]
    packet = truth
    output = [y0]
    t = times[0]
    y, velocity = y0, v0
    for target in times[1:]:
        count = max(1, int(math.ceil((target - t) / 0.025)))
        step = (target - t) / count
        for _ in range(count):
            def slope(tt: float, yy: float, vv: float) -> tuple[float, float]:
                accel = (
                    drive(tt, phase, packet)
                    - damping * vv
                    - stiffness * yy
                    - cubic * yy * yy * yy
                )
                return vv, accel

            k1y, k1v = slope(t, y, velocity)
            k2y, k2v = slope(t + step / 2, y + step * k1y / 2, velocity + step * k1v / 2)
            k3y, k3v = slope(t + step / 2, y + step * k2y / 2, velocity + step * k2v / 2)
            k4y, k4v = slope(t + step, y + step * k3y, velocity + step * k3v)
            y += step * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
            velocity += step * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            t += step
        output.append(y)
    return output


def truth_packet() -> dict:
    return {
        "omega": 1.37,
        "drive": {"a0": 0.18, "a1": 0.53, "a2": 0.16},
        "parameters": [0.34, 1.95, 0.27, 0.41],
        "states": [(-0.72, 0.21), (0.46, -0.34), (-0.12, 0.58), (0.31, -0.22)],
    }


def payload(case_id: str, samples: int = 10, segments: int = 3) -> dict:
    truth = truth_packet()
    data = {
        "case_id": case_id,
        "max_points": MAX_POINTS,
        "omega": truth["omega"],
        "drive": truth["drive"],
        "segments": [],
    }
    for index in range(segments):
        y0, v0 = truth["states"][index % len(truth["states"])]
        times = [round(0.22 * point, 3) for point in range(samples)]
        values = integrate(times, y0, v0, truth)
        data["segments"].append(
            {"times": times, "observations": [round(value, 6) for value in values]}
        )
    return data


def canonical(payload_value: dict) -> bytes:
    return (
        json.dumps(payload_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def execute(
    tmp_path: Path,
    data: dict | bytes,
    name: str,
    *,
    old_report: bytes | None = None,
    old_ledger: bytes | None = None,
    report_is_directory: bool = False,
) -> dict:
    folder = tmp_path / name
    folder.mkdir()
    source = folder / "input.json"
    report = folder / "report.json"
    ledger = folder / "ledger.jsonl"
    source.write_bytes(data if isinstance(data, bytes) else canonical(data))
    if old_report is not None:
        report.write_bytes(old_report)
    if old_ledger is not None:
        ledger.write_bytes(old_ledger)
    if report_is_directory:
        report.mkdir()
    process = subprocess.run(
        [sys.executable, CANDIDATE, str(source), str(report), str(ledger)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = {
        "process": process,
        "folder": folder,
        "source": source,
        "report": report,
        "ledger": ledger,
    }
    return result


def report_of(case: dict) -> dict:
    return json.loads(case["report"].read_text(encoding="utf-8"))


def ledger_rows(case: dict) -> list[dict]:
    path: Path = case["ledger"]
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_joint_recovery_returns_shared_parameters(tmp_path):
    """题面：「求解必须联合使用三段观测，不能逐段独立猜初态。」"""
    case = execute(tmp_path, payload("shared-parameters"), "case_01")
    report = report_of(case)
    assert abs(report["parameters"]["c"] - 0.34) < 2e-4
    assert abs(report["parameters"]["k"] - 1.95) < 2e-4


def test_recovery_recovers_cubic_and_phase(tmp_path):
    """题面：「共享参数需落在 `0.05 <= c <= 2.5`、`0.8 <= k <= 4.0`、`-0.5 <= beta <= 0.8`、`-pi <= phase <= pi`。」"""
    report = report_of(execute(tmp_path, payload("cubic-phase"), "case_02"))
    assert abs(report["parameters"]["beta"] - 0.27) < 2e-4
    assert abs(report["parameters"]["phase"] - 0.41) < 2e-4


def test_initial_states_follow_segment_order(tmp_path):
    """题面：「`initial_states` 按输入段序排列」"""
    report = report_of(execute(tmp_path, payload("state-order"), "case_03"))
    actual = [(item["segment_index"], item["y"], item["v"]) for item in report["initial_states"]]
    assert [item[0] for item in actual] == [0, 1, 2]
    assert abs(actual[0][1] + 0.72) < 2e-4
    assert abs(actual[2][2] - 0.58) < 2e-4


def test_reported_rms_is_recomputed_from_observations(tmp_path):
    """题面：「恢复后的 `rms` 必须由全部公开观测点复算，不能填常量。」"""
    case = execute(tmp_path, payload("residual-quality"), "case_04")
    report = report_of(case)
    assert report["quality"]["rms"] <= 0.002
    assert report["quality"]["jacobian_rank"] == 10


def test_two_segment_case_reports_rank_eight(tmp_path):
    """二段时参数总数为 4+2*2，雅可比满秩必须是 8。"""
    case = execute(tmp_path, payload("two-segment-rank", samples=16, segments=2), "two")
    report = report_of(case)
    assert case["process"].returncode == 0
    assert len(report["initial_states"]) == 2
    assert report["quality"]["jacobian_rank"] == 8
    assert report["quality"]["rms"] <= 0.002


def test_success_report_is_compact_sorted_utf8_with_newline(tmp_path):
    """题面：「成功报告使用 UTF-8、键排序、逗号后无空白，并在末尾保留一个换行。」"""
    case = execute(tmp_path, payload("canonical-output"), "case_05")
    raw = case["report"].read_bytes()
    assert raw.endswith(b"\n")
    assert b": " not in raw
    assert raw == canonical(json.loads(raw))


def test_success_writes_only_public_stdout_token(tmp_path):
    """题面：「成功时标准输出只写一行 `ODE_RECOVERED`，标准错误为空。」"""
    case = execute(tmp_path, payload("success-channel"), "case_06")
    assert case["process"].returncode == 0
    assert case["process"].stdout == "ODE_RECOVERED\n"
    assert case["process"].stderr == ""


def test_success_appends_one_ledger_record(tmp_path):
    """题面：「成功报告写完并完成账本追加后才算完成。」"""
    case = execute(tmp_path, payload("ledger-append"), "case_07")
    rows = ledger_rows(case)
    assert len(rows) == 1
    assert set(rows[0]) == {"case_id", "input_sha256", "output_sha256", "sequence"}
    assert rows[0]["sequence"] == 1


def test_identical_rerun_is_idempotent_for_report_and_ledger(tmp_path):
    """题面：「相同 `case_id`、相同输入摘要和相同报告摘要的重跑是幂等操作，不得追加重复记录」"""
    case = execute(tmp_path, payload("ledger-idempotent"), "case_08")
    first_report = case["report"].read_bytes()
    first_ledger = case["ledger"].read_bytes()
    again = subprocess.run(
        [sys.executable, CANDIDATE, str(case["source"]), str(case["report"]), str(case["ledger"])],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert again.returncode == 0
    assert case["report"].read_bytes() == first_report
    assert case["ledger"].read_bytes() == first_ledger


def test_changed_input_appends_with_next_sequence(tmp_path):
    """题面：「新的报告摘要必须追加为新记录；`sequence` 从 1 开始连续递增。」"""
    case = execute(tmp_path, payload("ledger-sequence", samples=9), "case_09")
    second = case["folder"] / "second.json"
    second.write_bytes(canonical(payload("ledger-sequence", samples=10)))
    process = subprocess.run(
        [sys.executable, CANDIDATE, str(second), str(case["report"]), str(case["ledger"])],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0
    assert [row["sequence"] for row in ledger_rows(case)] == [1, 2]


def test_refusal_preserves_old_report_and_ledger_bytes(tmp_path):
    """题面：「进程异常、输入拒收或发布失败时，账本必须回滚到最后一条完整记录，已有报告的旧字节保持不变。」"""
    old_report = b'{"old":"report"}\n'
    old_ledger = canonical(
        {"case_id": "old", "input_sha256": "0" * 64, "output_sha256": "1" * 64, "sequence": 1}
    )
    bad = payload("preserve")
    bad["segments"][0]["times"][2] = bad["segments"][0]["times"][1]
    case = execute(
        tmp_path,
        bad,
        "case_10",
        old_report=old_report,
        old_ledger=old_ledger,
    )
    assert case["process"].returncode == 2
    assert case["report"].read_bytes() == old_report
    assert case["ledger"].read_bytes() == old_ledger


def test_concurrent_append_keeps_one_well_formed_record(tmp_path):
    """题面：「账本追加需要独占锁。」"""
    folder = tmp_path / "case_11"
    folder.mkdir()
    source = folder / "input.json"
    report = folder / "report.json"
    ledger = folder / "ledger.jsonl"
    source.write_bytes(canonical(payload("ledger-lock")))
    processes = [
        subprocess.Popen(
            [sys.executable, CANDIDATE, str(source), str(report), str(ledger)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    completed = [process.communicate(timeout=90) for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert all(stdout == "ODE_RECOVERED\n" for stdout, _ in completed)
    assert len(ledger_rows({"ledger": ledger})) == 1


def test_incomplete_ledger_tail_is_recovered_from_checkpoint(tmp_path):
    """题面：「若上次进程在写入尾部时中断，下一次运行应丢弃不完整的末行，从最后一个换行形成的检查点恢复」"""
    old = canonical(
        {"case_id": "prior", "input_sha256": "a" * 64, "output_sha256": "b" * 64, "sequence": 1}
    )
    case = execute(
        tmp_path,
        payload("ledger-checkpoint"),
        "case_12",
        old_ledger=old + b'{"partial":',
    )
    rows = ledger_rows(case)
    assert [row["sequence"] for row in rows] == [1, 2]
    assert case["ledger"].read_bytes().endswith(b"\n")


def test_drive_object_requires_exact_keys(tmp_path):
    """题面：「`drive` 的精确键为 `a0`、`a1`、`a2`」"""
    bad = payload("drive-keys")
    bad["drive"]["extra"] = 0.1
    case = execute(tmp_path, bad, "case_13")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert case["process"].stderr.startswith("ODE_REFUSED: ")


def test_segment_count_two_and_five_are_separated(tmp_path):
    """题面：「`segments` 是二到四段的数组，每段精确键为 `times` 和 `observations`」"""
    accepted = execute(tmp_path, payload("segments-two", samples=6, segments=2), "case_14")
    refused = execute(tmp_path, payload("segments-five", samples=4, segments=5), "case_15")
    assert not refused["report"].exists()
    assert not refused["ledger"].exists()
    assert accepted["process"].returncode == 0
    assert refused["process"].returncode == 2


def test_times_must_be_strictly_increasing(tmp_path):
    """题面：「每段 `times` 从零开始并严格递增。」"""
    bad = payload("time-order")
    bad["segments"][0]["times"][3] = bad["segments"][0]["times"][2]
    case = execute(tmp_path, bad, "case_16")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "times must increase" in case["process"].stderr


def test_max_points_limit_of_48_is_supported(tmp_path):
    """题面：「所有段长度之和等于 48 时仍应运行，超过 48 时程序应拒绝。」"""
    case = execute(tmp_path, payload("points-limit", samples=16, segments=3), "case_17")
    assert case["process"].returncode == 0
    assert case["report"].is_file()


def test_total_observation_budget_above_48_is_refused(tmp_path):
    """题面：「`max_points` 必须等于允许的观测点上限 48」"""
    bad = payload("points-over", samples=13, segments=4)
    case = execute(tmp_path, bad, "case_18")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "observation budget exceeded" in case["process"].stderr


def test_case_id_rejects_path_like_characters(tmp_path):
    """题面：「交接编号限定为 1 到 48 个字母、数字、连字符或下划线字符。」"""
    case = execute(tmp_path, payload("../case"), "case_19")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert case["process"].stdout == ""


def test_rank_deficient_identical_segments_are_refused(tmp_path):
    """题面：「任何越界或不可辨识配置都必须拒收，并保留原有报告字节和账本字节。」"""
    duplicate = payload("rank-deficient", samples=7, segments=4)
    duplicate["segments"][1] = json.loads(json.dumps(duplicate["segments"][0]))
    duplicate["segments"][2] = json.loads(json.dumps(duplicate["segments"][0]))
    duplicate["segments"][3] = json.loads(json.dumps(duplicate["segments"][0]))
    case = execute(tmp_path, duplicate, "case_20")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert not case["report"].exists()


def test_refusal_uses_only_stderr_and_exit_two(tmp_path):
    """题面：「拒收时标准输出为空，标准错误只写一行，格式为 `ODE_REFUSED: <原因>`，进程退出码为 2。」"""
    case = execute(tmp_path, b"{not-json", "case_21")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert case["process"].stdout == ""
    assert case["process"].stderr.startswith("ODE_REFUSED: ")
    assert case["process"].stderr.count("\n") == 1


def test_report_top_level_key_set_is_exact(tmp_path):
    """题面：「报告精确顶层键为 `status`、`parameters`、`initial_states`、`quality`。」"""
    report = report_of(execute(tmp_path, payload("report-keys"), "case_22"))
    assert set(report) == {"status", "parameters", "initial_states", "quality"}
    assert report["status"] == "recovered"


def test_parameter_and_quality_key_sets_are_exact(tmp_path):
    """题面：「`parameters` 精确键为 `c`、`k`、`beta`、`phase`。」"""
    report = report_of(execute(tmp_path, payload("nested-keys"), "case_23"))
    assert set(report["parameters"]) == {"c", "k", "beta", "phase"}
    assert set(report["quality"]) == {"rms", "iterations", "jacobian_rank"}


def test_initial_state_entries_have_exact_keys(tmp_path):
    """题面：「每项精确键为 `segment_index`、`y`、`v`」"""
    report = report_of(execute(tmp_path, payload("state-keys"), "case_24"))
    assert all(set(item) == {"segment_index", "y", "v"} for item in report["initial_states"])
    assert all(isinstance(item["segment_index"], int) for item in report["initial_states"])


def test_duplicate_json_key_is_refused(tmp_path):
    """题面：「读取端的五个顶层成员分别是交接编号、观测预算、角频率、驱动包和轨迹段；键集合之外的内容一律视为错误。」"""
    raw = canonical(payload("duplicate-key"))[:-2] + b',"case_id":"again"}\n'
    case = execute(tmp_path, raw, "case_25")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "duplicate key" in case["process"].stderr


def test_top_level_extra_key_is_refused(tmp_path):
    """题面：「读取端的五个顶层成员分别是交接编号、观测预算、角频率、驱动包和轨迹段；键集合之外的内容一律视为错误。」"""
    bad = payload("top-extra")
    bad["comment"] = "not allowed"
    case = execute(tmp_path, bad, "case_26")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "extra keys" in case["process"].stderr


def test_segment_array_lengths_must_match(tmp_path):
    """题面：「两个数组等长且各有四到十六个有限数值。」"""
    bad = payload("length-mismatch")
    bad["segments"][0]["observations"].pop()
    case = execute(tmp_path, bad, "case_27")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "lengths differ" in case["process"].stderr


def test_non_numeric_observation_is_refused(tmp_path):
    """题面：「`drive` 的精确键为 `a0`、`a1`、`a2`，值都是有限数值。」"""
    bad = payload("non-numeric")
    bad["drive"]["a1"] = "0.53"
    case = execute(tmp_path, bad, "case_28")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "must be numeric" in case["process"].stderr


def test_publish_failure_keeps_existing_directory_and_ledger(tmp_path):
    """题面：「报告写出采用同目录临时文件、`fsync` 与原子替换，失败时不得留下半成品。」"""
    old_ledger = canonical(
        {"case_id": "prior", "input_sha256": "c" * 64, "output_sha256": "d" * 64, "sequence": 1}
    )
    case = execute(
        tmp_path,
        payload("publish-failure"),
        "case_29",
        old_ledger=old_ledger,
        report_is_directory=True,
    )
    leftovers = [path.name for path in case["folder"].iterdir() if path.name.startswith(".report.json.")]
    assert case["process"].returncode == 2
    assert case["report"].is_dir()
    assert case["ledger"].read_bytes() == old_ledger
    assert leftovers == []


def test_wrong_max_points_value_is_refused(tmp_path):
    """题面：「`max_points` 必须等于允许的观测点上限 48」"""
    bad = payload("wrong-budget-name")
    bad["max_points"] = 47
    case = execute(tmp_path, bad, "case_30")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "max_points must equal 48" in case["process"].stderr


def test_candidate_has_required_entry_and_runs(tmp_path):
    """题面：「调用方式是 `python3 /app/ode_recover.py <input.json> <report.json> <ledger.jsonl>`。」"""
    case = execute(tmp_path, payload("entry-runnable"), "case_31")
    assert Path("/app/ode_recover.py").as_posix() in "/app/ode_recover.py"
    assert os.environ.get("ODE_ENTRY", "/app/ode_recover.py") == CANDIDATE
    assert case["process"].returncode == 0
