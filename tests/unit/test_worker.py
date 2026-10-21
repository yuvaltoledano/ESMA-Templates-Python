"""Tests for the background-job worker (esma_milan.api.worker).

`process_one()` drives a job to `done` on the synthetic fixture in both
modes; a job whose pipeline raises ends `failed` with the same `error`
code the synchronous endpoint returns for the same input; the Excel
row guard ends a job `failed` / `output_too_large`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import esma_milan.runner as runner_module
from esma_milan.api import jobs, worker

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"
LOANS = (SYNTHETIC / "loans.csv").read_bytes()
COLLATERALS = (SYNTHETIC / "collaterals.csv").read_bytes()


def _make_job(
    root: Path,
    *,
    mode: str = "workbook",
    loans: bytes = LOANS,
    collaterals: bytes = COLLATERALS,
    deal: str = "SYNTHETIC_FIXTURE",
) -> jobs.Job:
    request = jobs.JobRequest(
        deal_name=deal,
        mode=mode,  # type: ignore[arg-type]
        aggregation="auto",
        min_coverage=0.85,
        created_at=jobs.now_iso(),
    )
    job = jobs.begin_job(root, request)
    (job.inputs_dir / "loans.csv").write_bytes(loans)
    (job.inputs_dir / "collaterals.csv").write_bytes(collaterals)
    return jobs.commit_job(root, job)


def test_process_one_returns_none_on_empty_queue(tmp_path: Path) -> None:
    assert worker.process_one(tmp_path, "w") is None


def test_workbook_job_runs_to_done(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    status = worker.process_one(tmp_path, "w")
    assert status is not None
    assert status.state == "done"
    assert status.stage == jobs.JobStage(11, 11, "Writing workbook")
    assert status.result is not None
    assert status.result["filename"] == "2024-06-30 SYNTHETIC_FIXTURE Flattened loans and collaterals.xlsx"
    out = job.output_dir / status.result["filename"]
    assert out.is_file() and out.stat().st_size == status.result["size_bytes"]
    assert not job.inputs_dir.exists(), "inputs must be deleted when the job finishes"
    assert not (job.path / "output.tmp").exists(), "no partial output left behind"
    assert not list(job.path.glob("running.*"))


def test_analysis_job_runs_to_done_and_writes_analysis_json(tmp_path: Path) -> None:
    job = _make_job(tmp_path, mode="analysis")
    status = worker.process_one(tmp_path, "w")
    assert status is not None and status.state == "done"
    assert status.stage is not None and status.stage.index == 9  # dry run stops before the writer
    body = json.loads((job.output_dir / "analysis.json").read_text())
    assert set(body) == {"deal_name", "summary", "stratifications", "execution_summary"}
    assert body["summary"]["loan_count"] == 8
    assert not job.inputs_dir.exists()


def test_bad_csv_job_fails_with_same_code_as_sync_endpoint(tmp_path: Path) -> None:
    """The garbage bytes from test_api.py's malformed-CSV case: the
    synchronous endpoint classifies them as 400 `invalid_csv`; the job
    must carry the same code."""
    job = _make_job(tmp_path, loans=b"\x00\x01\x02 not,a,csv \xff\xfe")
    status = worker.process_one(tmp_path, "w")
    assert status is not None and status.state == "failed"
    assert status.error is not None
    assert status.error["error"] == "invalid_csv"
    assert status.error["message"] == "The uploaded data could not be processed."
    assert "reason" in (status.error["details"] or {})
    assert not job.inputs_dir.exists()
    assert not (job.output_dir / "analysis.json").exists()


def test_row_limit_guard_fails_job_with_output_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the Excel limit so the 8-loan fixture trips it; the job must
    end `output_too_large` naming the sheet, with no output written."""
    monkeypatch.setattr(runner_module, "EXCEL_MAX_ROWS", 5)
    job = _make_job(tmp_path)
    status = worker.process_one(tmp_path, "w")
    assert status is not None and status.state == "failed"
    assert status.error is not None
    assert status.error["error"] == "output_too_large"
    assert status.error["details"]["limit"] == 5
    assert status.error["details"]["sheet"]  # names the first sheet over the limit
    assert not any(job.output_dir.iterdir()) if job.output_dir.exists() else True
    assert not (job.path / "output.tmp").exists()


def test_run_job_records_progress_in_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stage boundary is written to status.json as it happens."""
    seen: list[tuple[int, int, str]] = []
    real_update = jobs.update_stage

    def spy(job: jobs.Job, index: int, total: int, label: str) -> None:
        seen.append((index, total, label))
        real_update(job, index, total, label)

    monkeypatch.setattr(jobs, "update_stage", spy)
    _make_job(tmp_path)
    status = worker.process_one(tmp_path, "w")
    assert status is not None and status.state == "done"
    assert [i for i, _, _ in seen] == list(range(1, 12))
    assert all(t == 11 for _, t, _ in seen)
    assert seen[-1][2] == "Writing workbook"


def test_startup_recovers_and_sweeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jobs.RETENTION_ENV, "12")
    job = _make_job(tmp_path)
    claimed = jobs.claim_next(tmp_path, "dead")
    assert claimed is not None
    worker.startup(tmp_path)
    assert job.read_status().state == "failed"
    assert job.read_status().error == {
        "error": "interrupted",
        "message": job.read_status().error["message"],  # type: ignore[index]
        "details": {"stage": None},
    }
