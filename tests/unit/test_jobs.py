"""Tests for the on-disk job store (esma_milan.api.jobs).

Covers: id validation before any filesystem access; atomic status
writes; exclusive claiming (two claimers, one wins); the sweep deleting
expired jobs and keeping live ones; crash recovery marking `running`
jobs `failed` / `interrupted` with no retry.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from esma_milan.api import jobs


def _request(deal: str = "DEAL", mode: str = "workbook", created_at: str | None = None) -> jobs.JobRequest:
    return jobs.JobRequest(
        deal_name=deal,
        mode=mode,  # type: ignore[arg-type]
        aggregation="auto",
        min_coverage=0.85,
        created_at=created_at or jobs.now_iso(),
        filenames={"loans": "loans.csv", "collaterals": "collaterals.csv"},
    )


def _queued_job(root: Path, **kw: object) -> jobs.Job:
    job = jobs.begin_job(root, _request(**kw))  # type: ignore[arg-type]
    (job.inputs_dir / "loans.csv").write_text("a,b\n1,2\n")
    (job.inputs_dir / "collaterals.csv").write_text("a,b\n1,2\n")
    return jobs.commit_job(root, job)


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../etc/passwd",
        "..",
        "ABCDEF0123456789ABCDEF0123456789",  # uppercase
        "0123456789abcdef0123456789abcde",  # 31 chars
        "0123456789abcdef0123456789abcdef0",  # 33 chars
        "0123456789abcdef0123456789abcdeg",  # non-hex
        "0123456789abcdef0123456789abcdef\n",
    ],
)
def test_validate_job_id_rejects_anything_but_32_lowercase_hex(bad: str) -> None:
    with pytest.raises(jobs.JobNotFoundError):
        jobs.validate_job_id(bad)


def test_validate_job_id_accepts_uuid4_hex() -> None:
    job_id = jobs.new_job_id()
    assert jobs.validate_job_id(job_id) == job_id


def test_get_job_rejects_bad_id_without_touching_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A traversal-shaped id must be refused by the regex alone."""
    touched: list[str] = []
    real_is_dir = Path.is_dir

    def spy(self: Path) -> bool:
        touched.append(str(self))
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", spy)
    with pytest.raises(jobs.JobNotFoundError):
        jobs.get_job(tmp_path, "../../etc")
    assert touched == []


def test_get_job_unknown_but_well_formed_id_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(jobs.JobNotFoundError):
        jobs.get_job(tmp_path, jobs.new_job_id())


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_write_json_atomic_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "status.json"
    jobs.write_json_atomic(target, {"a": 1})
    jobs.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.json"]


def test_write_json_atomic_failure_keeps_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the replace step fails, the previous content is intact and the
    temp file is gone: a reader can never observe a partial file."""
    target = tmp_path / "status.json"
    jobs.write_json_atomic(target, {"state": "queued"})

    def boom(src: object, dst: object) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk went away"):
        jobs.write_json_atomic(target, {"state": "running"})
    assert json.loads(target.read_text()) == {"state": "queued"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.json"]


# ---------------------------------------------------------------------------
# Creation and claiming
# ---------------------------------------------------------------------------


def test_begin_commit_creates_queued_job_with_marker(tmp_path: Path) -> None:
    job = _queued_job(tmp_path)
    assert job.path == tmp_path / job.job_id
    assert job.read_status().state == "queued"
    assert job.queued_marker.exists()
    assert job.read_request().deal_name == "DEAL"
    assert not list(tmp_path.glob("*.tmp"))


def test_uncommitted_job_is_invisible_and_abortable(tmp_path: Path) -> None:
    job = jobs.begin_job(tmp_path, _request())
    assert list(jobs.iter_jobs(tmp_path)) == []
    with pytest.raises(jobs.JobNotFoundError):
        jobs.get_job(tmp_path, job.job_id)
    jobs.abort_job(job)
    assert not job.path.exists()


def test_claim_is_exclusive_between_two_claimers(tmp_path: Path) -> None:
    job = _queued_job(tmp_path)
    first = jobs.claim_next(tmp_path, "worker-a")
    second = jobs.claim_next(tmp_path, "worker-b")
    assert first is not None and first.job_id == job.job_id
    assert second is None
    assert first.read_status().state == "running"
    assert (first.path / "running.worker-a").exists()
    assert not first.queued_marker.exists()


def test_claim_under_concurrency_hands_every_job_out_exactly_once(tmp_path: Path) -> None:
    n = 24
    for i in range(n):
        _queued_job(tmp_path, created_at=f"2026-01-01T00:00:{i:02d}+00:00")
    claimed: dict[str, list[str]] = {"a": [], "b": [], "c": []}

    def drain(worker: str) -> None:
        while True:
            job = jobs.claim_next(tmp_path, worker)
            if job is None:
                return
            claimed[worker].append(job.job_id)

    threads = [threading.Thread(target=drain, args=(w,)) for w in claimed]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    all_ids = [jid for ids in claimed.values() for jid in ids]
    assert len(all_ids) == n
    assert len(set(all_ids)) == n


def test_claim_takes_oldest_first(tmp_path: Path) -> None:
    newer = _queued_job(tmp_path, created_at="2026-01-02T00:00:00+00:00")
    older = _queued_job(tmp_path, created_at="2026-01-01T00:00:00+00:00")
    first = jobs.claim_next(tmp_path, "w")
    assert first is not None and first.job_id == older.job_id
    second = jobs.claim_next(tmp_path, "w")
    assert second is not None and second.job_id == newer.job_id


# ---------------------------------------------------------------------------
# Finishing
# ---------------------------------------------------------------------------


def test_finish_job_deletes_inputs_and_marker(tmp_path: Path) -> None:
    job = _queued_job(tmp_path)
    claimed = jobs.claim_next(tmp_path, "w")
    assert claimed is not None
    jobs.update_stage(claimed, 3, 11, "Stage 3: identifiers")
    assert claimed.read_status().stage == jobs.JobStage(3, 11, "Stage 3: identifiers")
    status = jobs.finish_job(claimed, state="done", result={"filename": "x.xlsx", "size_bytes": 1})
    assert status.state == "done" and status.finished_at is not None
    assert not job.inputs_dir.exists()
    assert not list(job.path.glob("running.*"))
    assert job.read_status().result == {"filename": "x.xlsx", "size_bytes": 1}


# ---------------------------------------------------------------------------
# Recovery and sweep
# ---------------------------------------------------------------------------


def test_recover_marks_running_jobs_failed_interrupted_without_retry(tmp_path: Path) -> None:
    job = _queued_job(tmp_path)
    claimed = jobs.claim_next(tmp_path, "dead-worker")
    assert claimed is not None
    jobs.update_stage(claimed, 7, 11, "Stage 7: flatten and derived fields")
    untouched = _queued_job(tmp_path)

    failed = jobs.recover_interrupted(tmp_path)

    assert failed == [job.job_id]
    status = job.read_status()
    assert status.state == "failed"
    assert status.error is not None
    assert status.error["error"] == "interrupted"
    assert "restarted" in status.error["message"] and "submit" in status.error["message"]
    assert status.error["details"] == {"stage": {"index": 7, "total": 11, "label": "Stage 7: flatten and derived fields"}}
    assert not job.inputs_dir.exists()
    # Not re-queued: a second recover finds nothing, and the job cannot be claimed.
    assert jobs.recover_interrupted(tmp_path) == []
    assert untouched.read_status().state == "queued"
    next_claim = jobs.claim_next(tmp_path, "w")
    assert next_claim is not None and next_claim.job_id == untouched.job_id


def test_sweep_deletes_expired_and_keeps_live_and_running(tmp_path: Path) -> None:
    """Jobs are created and claimed one at a time (claim_next takes the
    oldest queued job, so each claim must see exactly one candidate)."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    keep = timedelta(hours=12)

    def _finished(finished_at: datetime) -> jobs.Job:
        job = _queued_job(tmp_path)
        claimed = jobs.claim_next(tmp_path, "w")
        assert claimed is not None and claimed.job_id == job.job_id
        status = jobs.finish_job(claimed, state="done", result={"filename": "x", "size_bytes": 1})
        status.finished_at = finished_at.isoformat(timespec="seconds")
        job.write_status(status)
        return job

    expired_done = _finished(now - timedelta(hours=13))
    fresh_done = _finished(now - timedelta(hours=1))

    running = _queued_job(tmp_path, created_at=(now - timedelta(hours=40)).isoformat(timespec="seconds"))
    claimed = jobs.claim_next(tmp_path, "w")
    assert claimed is not None and claimed.job_id == running.job_id

    expired_never_finished = _queued_job(
        tmp_path, created_at=(now - timedelta(hours=30)).isoformat(timespec="seconds")
    )
    live_queued = _queued_job(tmp_path, created_at=(now - timedelta(minutes=5)).isoformat(timespec="seconds"))

    stale_tmp = tmp_path / f"{jobs.new_job_id()}.tmp"
    stale_tmp.mkdir()
    old = (now - timedelta(hours=2)).timestamp()
    os.utime(stale_tmp, (old, old))

    removed = jobs.sweep(tmp_path, keep, now=now)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert removed == 3
    assert expired_done.job_id not in remaining
    assert expired_never_finished.job_id not in remaining
    assert stale_tmp.name not in remaining
    assert fresh_done.job_id in remaining
    assert running.job_id in remaining
    assert live_queued.job_id in remaining


def test_config_reads_environment_per_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(jobs.JOBS_DIR_ENV, str(tmp_path / "j"))
    monkeypatch.setenv(jobs.RETENTION_ENV, "2.5")
    monkeypatch.setenv(jobs.MAX_UPLOAD_MB_ENV, "600")
    assert jobs.jobs_root() == tmp_path / "j"
    assert jobs.retention() == timedelta(hours=2.5)
    assert jobs.max_upload_bytes() == 600 * 1024 * 1024
    monkeypatch.delenv(jobs.JOBS_DIR_ENV)
    monkeypatch.delenv(jobs.RETENTION_ENV)
    monkeypatch.delenv(jobs.MAX_UPLOAD_MB_ENV)
    assert jobs.jobs_root() == Path("data/jobs")
    assert jobs.retention() == timedelta(hours=12)
    assert jobs.max_upload_mb() == 50
