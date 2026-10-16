"""Background-job worker: ``esma-milan-worker``.

A single process that drains the on-disk job store (``api/jobs.py``):

    recover_interrupted()          # jobs left `running` by a dead worker
    loop:
        sweep() once a minute      # delete expired jobs
        job = claim_next()         # oldest queued job, exclusive claim
        run_job(job) if job else sleep(IDLE_SLEEP_SECONDS)

One job at a time by construction: the loop is sequential and the
deployment runs one worker container. A job runs the same
``runner.run_pipeline`` the synchronous endpoint runs, with
``on_progress`` publishing each stage boundary to ``status.json``, and
every failure classified by ``handlers.classify_pipeline_error`` so the
job carries the same ``error`` code and message ``POST /api/process``
would have returned for the same input.

Outputs are never left half-written: the workbook is produced under
``output.tmp/`` and renamed into ``output/`` only on success; the analysis
JSON is written atomically.

``SIGTERM`` (``docker compose stop``) sets a flag checked between jobs,
so a job in flight finishes if the stop grace period allows; otherwise
the next worker's ``recover_interrupted`` marks it ``failed`` /
``interrupted`` and the pool must be resubmitted.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from esma_milan.api import jobs
from esma_milan.api.handlers import (
    ApiError,
    analysis_response_from_result,
    build_analysis_result,
    classify_pipeline_error,
    default_taxonomy_path,
)
from esma_milan.runner import ProgressCallback, run_pipeline

log = structlog.get_logger(__name__)

IDLE_SLEEP_SECONDS = 1.5
SWEEP_INTERVAL_SECONDS = 60.0
ANALYSIS_FILENAME = "analysis.json"

_stop_requested = False


def _request_stop(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    log.info("worker_stop_requested", signal=signum)


# ---------------------------------------------------------------------------
# Running one job
# ---------------------------------------------------------------------------


def run_job(job: jobs.Job) -> jobs.JobStatus:
    """Run a claimed job to ``done`` or ``failed`` and return its final status.

    Never raises for a pipeline or input problem: those become the job's
    ``error``. A failure to record the outcome (e.g. the jobs volume is
    gone) does propagate, since there is nothing sensible to record.
    """
    request = job.read_request()
    started = time.perf_counter()
    log.info(
        "job_start",
        job_id=job.job_id,
        deal_name=request.deal_name,
        mode=request.mode,
    )

    def on_progress(index: int, total: int, label: str) -> None:
        jobs.update_stage(job, index, total, label)

    try:
        result = _execute(job, request, on_progress)
    except ApiError as exc:
        return _fail(job, request, exc, started)
    except Exception as exc:
        return _fail(job, request, classify_pipeline_error(exc, deal_name=request.deal_name), started)

    status = jobs.finish_job(job, state="done", result=result)
    log.info(
        "job_done",
        job_id=job.job_id,
        deal_name=request.deal_name,
        mode=request.mode,
        filename=result["filename"],
        size_bytes=result["size_bytes"],
        duration_s=round(time.perf_counter() - started, 1),
    )
    return status


def _fail(
    job: jobs.Job, request: jobs.JobRequest, error: ApiError, started: float
) -> jobs.JobStatus:
    shutil.rmtree(job.path / "output.tmp", ignore_errors=True)
    status = jobs.finish_job(
        job,
        state="failed",
        error={"error": error.error, "message": error.message, "details": error.details},
    )
    log.warning(
        "job_failed",
        job_id=job.job_id,
        deal_name=request.deal_name,
        mode=request.mode,
        error=error.error,
        duration_s=round(time.perf_counter() - started, 1),
    )
    return status


def _execute(
    job: jobs.Job,
    request: jobs.JobRequest,
    on_progress: ProgressCallback,
) -> dict[str, Any]:
    """Run the pipeline for `job`; return the `result` record for status.json."""
    loans_path = job.inputs_dir / "loans.csv"
    collaterals_path = job.inputs_dir / "collaterals.csv"
    uploaded_taxonomy = job.inputs_dir / "taxonomy.xlsx"
    taxonomy_path = uploaded_taxonomy if uploaded_taxonomy.is_file() else default_taxonomy_path()
    aggregation_method = None if request.aggregation == "auto" else request.aggregation

    job.output_dir.mkdir(exist_ok=True)

    if request.mode == "analysis":
        pipeline = run_pipeline(
            loans_file_path=loans_path,
            collaterals_file_path=collaterals_path,
            taxonomy_file_path=taxonomy_path,
            deal_name=request.deal_name,
            output_dir=None,
            aggregation_method=aggregation_method,
            min_coverage=request.min_coverage,
            dry_run=True,
            verbose=True,
            on_progress=on_progress,
        )
        analysis = build_analysis_result(pipeline, request.deal_name, request.aggregation)
        body = analysis_response_from_result(analysis).model_dump()
        target = job.output_dir / ANALYSIS_FILENAME
        jobs.write_json_atomic(target, body)
        return {"filename": ANALYSIS_FILENAME, "size_bytes": target.stat().st_size}

    # Workbook mode: write under output.tmp/, then move the single file
    # into output/ so a reader never sees a partial workbook.
    tmp_out = job.path / "output.tmp"
    shutil.rmtree(tmp_out, ignore_errors=True)
    pipeline = run_pipeline(
        loans_file_path=loans_path,
        collaterals_file_path=collaterals_path,
        taxonomy_file_path=taxonomy_path,
        deal_name=request.deal_name,
        output_dir=tmp_out,
        aggregation_method=aggregation_method,
        min_coverage=request.min_coverage,
        dry_run=False,
        verbose=True,
        on_progress=on_progress,
    )
    produced = pipeline.output_path
    assert produced is not None, "run_pipeline(output_dir=...) must return output_path"
    final = job.output_dir / produced.name
    os.replace(produced, final)
    shutil.rmtree(tmp_out, ignore_errors=True)
    return {"filename": final.name, "size_bytes": final.stat().st_size}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def process_one(root: Path, worker: str) -> jobs.JobStatus | None:
    """Claim and run one job. Returns its final status, or None if the
    queue was empty. Callable from tests."""
    job = jobs.claim_next(root, worker)
    if job is None:
        return None
    return run_job(job)


def startup(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    interrupted = jobs.recover_interrupted(root)
    if interrupted:
        log.warning("worker_recovered_interrupted_jobs", job_ids=interrupted)
    removed = jobs.sweep(root, jobs.retention())
    log.info(
        "worker_started",
        jobs_dir=str(root),
        retention_hours=jobs.retention().total_seconds() / 3600,
        swept=removed,
    )


def main() -> None:
    """Entry point for ``esma-milan-worker``."""
    # Same console logging as the API process (no colours on a pipe).
    from esma_milan.api.server import _configure_logging

    _configure_logging()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    root = jobs.jobs_root()
    worker = jobs.worker_id()
    startup(root)

    last_sweep = time.monotonic()
    while not _stop_requested:
        if time.monotonic() - last_sweep >= SWEEP_INTERVAL_SECONDS:
            removed = jobs.sweep(root, jobs.retention())
            if removed:
                log.info("jobs_swept", removed=removed, at=datetime.now(UTC).isoformat())
            last_sweep = time.monotonic()
        try:
            status = process_one(root, worker)
        except Exception:
            # Recording the outcome failed (e.g. the jobs volume vanished).
            # Log and keep serving; the job is left for recover_interrupted.
            log.error("worker_loop_error", exc_info=True)
            status = None
        if status is None:
            time.sleep(IDLE_SLEEP_SECONDS)
    log.info("worker_stopped")
