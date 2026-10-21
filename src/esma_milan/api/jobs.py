"""On-disk job store for background pipeline runs.

A job is a directory under the jobs root (``ESMA_MILAN_JOBS_DIR``, default
``data/jobs``), named by a 32-hex ``uuid4``::

    <root>/<job_id>/
        request.json      what was asked (deal name, mode, options, filenames)
        status.json       the state machine below; replaced atomically
        inputs/           loans.csv, collaterals.csv; deleted the moment
                          the job finishes
        output/           the workbook, or analysis.json
        queued            marker file while queued ...
        running.<worker>  ... renamed by the worker that claimed the job

State machine (``status.json["state"]``)::

    queued -> running -> done
                      -> failed

Design points:

- **No shared memory.** Uvicorn workers and the worker process are
  separate processes (several containers in the deployment); the
  directory is the only shared state, so every reader must tolerate a
  writer replacing a file between two reads.
- **Atomic writes.** Every JSON file is written to a temporary file in the
  same directory and then ``os.replace``d, so a reader never sees a torn
  file. A job directory is built as ``<job_id>.tmp`` and renamed into
  place, so a lister never sees a half-written job.
- **Exclusive claim.** Claiming is an ``os.rename`` of the ``queued``
  marker to ``running.<worker-id>``; a rename is atomic on POSIX, so two
  workers can never claim the same job even though the deployment runs
  one.
- **No retry.** A job found ``running`` when a worker starts was
  interrupted by a restart (a redeploy, or the kernel killing the worker
  for memory); it becomes ``failed`` with code ``interrupted`` and the
  pool must be resubmitted. Retrying would usually repeat an
  out-of-memory kill and waste the run time.
- **Retention.** ``ESMA_MILAN_JOB_RETENTION_HOURS`` (default 12) after a
  job finishes (or was created, if it never finished) its directory is
  deleted by the worker's periodic sweep.

Ids are validated against ``JOB_ID_RE`` before any filesystem access, so
a request for ``../something`` never touches the disk.

This module has no FastAPI dependency: the API, the worker and the tests
all use it directly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

JOBS_DIR_ENV = "ESMA_MILAN_JOBS_DIR"
DEFAULT_JOBS_DIR = "data/jobs"
RETENTION_ENV = "ESMA_MILAN_JOB_RETENTION_HOURS"
DEFAULT_RETENTION_HOURS = 12.0
MAX_UPLOAD_MB_ENV = "ESMA_MILAN_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 50

# A `<job_id>.tmp` directory older than this was abandoned by an API
# process that died mid-upload; the sweep removes it.
STALE_TMP_AGE = timedelta(hours=1)

QUEUED_MARKER = "queued"
RUNNING_MARKER_PREFIX = "running."

JobState = Literal["queued", "running", "done", "failed"]
JobMode = Literal["workbook", "analysis"]


class JobNotFoundError(Exception):
    """The id is malformed or no such job directory exists."""


# ---------------------------------------------------------------------------
# Configuration (read per call so tests and deployments can set the env)
# ---------------------------------------------------------------------------


def jobs_root() -> Path:
    return Path(os.environ.get(JOBS_DIR_ENV, DEFAULT_JOBS_DIR))


def retention() -> timedelta:
    return timedelta(hours=float(os.environ.get(RETENTION_ENV, DEFAULT_RETENTION_HOURS)))


def max_upload_mb() -> int:
    return int(os.environ.get(MAX_UPLOAD_MB_ENV, DEFAULT_MAX_UPLOAD_MB))


def max_upload_bytes() -> int:
    return max_upload_mb() * 1024 * 1024


def worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """UTC timestamp, second precision, ISO 8601 with offset."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class JobRequest:
    """What ``POST /api/jobs`` was asked to do. Stored as request.json."""

    deal_name: str
    mode: JobMode
    aggregation: str  # "auto" | "by_loan" | "by_group"
    min_coverage: float
    created_at: str
    filenames: dict[str, str] = field(default_factory=dict)  # field -> original name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobRequest:
        return cls(
            deal_name=str(d["deal_name"]),
            mode=d["mode"],
            aggregation=str(d["aggregation"]),
            min_coverage=float(d["min_coverage"]),
            created_at=str(d["created_at"]),
            filenames=dict(d.get("filenames") or {}),
        )


@dataclass(frozen=True)
class JobStage:
    index: int
    total: int
    label: str


@dataclass
class JobStatus:
    """status.json. ``error`` is the ``ErrorResponse`` shape
    (``{error, message, details}``); ``result`` is
    ``{filename, size_bytes}``."""

    state: JobState
    mode: JobMode
    deal_name: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    stage: JobStage | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobStatus:
        stage = d.get("stage")
        return cls(
            state=d["state"],
            mode=d["mode"],
            deal_name=str(d["deal_name"]),
            created_at=str(d["created_at"]),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            stage=JobStage(**stage) if stage else None,
            error=d.get("error"),
            result=d.get("result"),
        )


@dataclass(frozen=True)
class Job:
    """A job directory. ``path`` is the final directory, or the
    ``<id>.tmp`` directory between ``begin_job`` and ``commit_job``."""

    job_id: str
    path: Path

    @property
    def request_path(self) -> Path:
        return self.path / "request.json"

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def inputs_dir(self) -> Path:
        return self.path / "inputs"

    @property
    def output_dir(self) -> Path:
        return self.path / "output"

    @property
    def queued_marker(self) -> Path:
        return self.path / QUEUED_MARKER

    def read_request(self) -> JobRequest:
        return JobRequest.from_dict(read_json(self.request_path))

    def read_status(self) -> JobStatus:
        return JobStatus.from_dict(read_json(self.status_path))

    def write_status(self, status: JobStatus) -> None:
        write_json_atomic(self.status_path, status.to_dict())


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, obj: dict[str, Any], *, sort_keys: bool = True) -> None:
    """Write ``obj`` as JSON to ``path`` via a temp file + ``os.replace``.

    A reader of ``path`` sees either the previous complete file or the
    new complete file, never a partial one. On any failure the temp file
    is removed and ``path`` is left as it was.

    Keys are sorted by default so ``status.json`` and ``request.json`` are
    deterministic. Pass ``sort_keys=False`` when the order carries
    meaning, as it does for the analysis result: its ``stratifications``
    are keyed in the registry order the page renders tiles in.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=sort_keys)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, retrying once if a writer is mid-replace on a
    filesystem where ``os.replace`` is not atomic."""
    for attempt in (1, 2):
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"{path} does not hold a JSON object")
            return data
        except json.JSONDecodeError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Ids and lookup
# ---------------------------------------------------------------------------


def new_job_id() -> str:
    return uuid.uuid4().hex


def validate_job_id(job_id: str) -> str:
    """Return ``job_id`` if it is 32 lowercase hex characters; raise
    ``JobNotFoundError`` otherwise. Called before any filesystem access."""
    if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
        raise JobNotFoundError(job_id)
    return job_id


def get_job(root: Path, job_id: str) -> Job:
    validate_job_id(job_id)
    path = root / job_id
    if not path.is_dir() or not (path / "status.json").is_file():
        raise JobNotFoundError(job_id)
    return Job(job_id=job_id, path=path)


def iter_jobs(root: Path) -> Iterator[Job]:
    """Every committed job directory, in no particular order."""
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry.is_dir() and JOB_ID_RE.fullmatch(entry.name) and (entry / "status.json").is_file():
            yield Job(job_id=entry.name, path=entry)


# ---------------------------------------------------------------------------
# Creation (API side)
# ---------------------------------------------------------------------------


def begin_job(root: Path, request: JobRequest) -> Job:
    """Create ``<root>/<id>.tmp`` with ``request.json`` and an empty
    ``inputs/``. The caller streams the uploads into ``inputs/`` and then
    calls ``commit_job`` (or ``abort_job`` on failure)."""
    root.mkdir(parents=True, exist_ok=True)
    job_id = new_job_id()
    tmp = root / f"{job_id}.tmp"
    tmp.mkdir()
    job = Job(job_id=job_id, path=tmp)
    job.inputs_dir.mkdir()
    write_json_atomic(job.request_path, request.to_dict())
    return job


def commit_job(root: Path, job: Job) -> Job:
    """Write the ``queued`` status and marker, then rename the directory
    into place. After this the worker can claim the job."""
    request = job.read_request()
    job.write_status(
        JobStatus(
            state="queued",
            mode=request.mode,
            deal_name=request.deal_name,
            created_at=request.created_at,
        )
    )
    job.queued_marker.touch()
    final = root / job.job_id
    os.rename(job.path, final)
    return Job(job_id=job.job_id, path=final)


def abort_job(job: Job) -> None:
    shutil.rmtree(job.path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def claim_next(root: Path, worker: str) -> Job | None:
    """Claim the oldest queued job by renaming its marker, or return None.

    Candidates are ordered by ``created_at``. The rename is the only
    arbitration: if it raises ``FileNotFoundError`` another worker won
    that job and the next candidate is tried.
    """
    candidates: list[tuple[str, Job]] = []
    for job in iter_jobs(root):
        if not job.queued_marker.exists():
            continue
        try:
            status = job.read_status()
        except (OSError, ValueError):
            continue
        if status.state == "queued":
            candidates.append((status.created_at, job))
    candidates.sort(key=lambda pair: pair[0])

    for _, job in candidates:
        try:
            os.rename(job.queued_marker, job.path / f"{RUNNING_MARKER_PREFIX}{worker}")
        except FileNotFoundError:
            continue
        status = job.read_status()
        status.state = "running"
        status.started_at = now_iso()
        job.write_status(status)
        return job
    return None


def update_stage(job: Job, index: int, total: int, label: str) -> None:
    status = job.read_status()
    status.stage = JobStage(index=index, total=total, label=label)
    job.write_status(status)


def finish_job(
    job: Job,
    *,
    state: Literal["done", "failed"],
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> JobStatus:
    """Record the terminal state, delete the inputs and the running marker."""
    shutil.rmtree(job.inputs_dir, ignore_errors=True)
    status = job.read_status()
    status.state = state
    status.finished_at = now_iso()
    status.result = result
    status.error = error
    job.write_status(status)
    for marker in job.path.glob(f"{RUNNING_MARKER_PREFIX}*"):
        marker.unlink(missing_ok=True)
    job.queued_marker.unlink(missing_ok=True)
    return status


def delete_job(job: Job) -> None:
    shutil.rmtree(job.path, ignore_errors=True)


def recover_interrupted(root: Path) -> list[str]:
    """At worker startup: every job still ``running`` was interrupted by
    the previous worker's death. Mark it ``failed`` / ``interrupted``.
    Returns the affected ids."""
    failed: list[str] = []
    for job in iter_jobs(root):
        try:
            status = job.read_status()
        except (OSError, ValueError):
            continue
        if status.state != "running":
            continue
        finish_job(
            job,
            state="failed",
            error={
                "error": "interrupted",
                "message": (
                    "The worker was restarted while this job was running (a "
                    "redeploy, or the server ran out of memory and the worker "
                    "was killed). The job was not retried; please submit the "
                    "pool again."
                ),
                "details": {"stage": asdict(status.stage) if status.stage else None},
            },
        )
        failed.append(job.job_id)
    return failed


def sweep(root: Path, keep_for: timedelta, now: datetime | None = None) -> int:
    """Delete job directories older than ``keep_for`` and abandoned
    ``<id>.tmp`` directories older than ``STALE_TMP_AGE``. Returns the
    number of directories removed."""
    if not root.is_dir():
        return 0
    current = now or datetime.now(UTC)
    removed = 0
    for job in iter_jobs(root):
        try:
            status = job.read_status()
        except (OSError, ValueError):
            continue
        if status.state == "running":
            continue
        anchor = status.finished_at or status.created_at
        try:
            when = parse_iso(anchor)
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        if current - when > keep_for:
            delete_job(job)
            removed += 1
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.endswith(".tmp"):
            age = current - datetime.fromtimestamp(entry.stat().st_mtime, tz=UTC)
            if age > STALE_TMP_AGE:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
    return removed
