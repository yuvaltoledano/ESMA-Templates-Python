"""Tests for the background-job endpoints (POST/GET/DELETE /api/jobs...).

The TestClient drives the API; the worker's `process_one` is called by
the test between requests to move a job through the state machine. The
jobs directory is `tmp_path` via ESMA_MILAN_JOBS_DIR.
"""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl
import pytest
from fastapi.testclient import TestClient

from esma_milan.api import jobs, worker
from esma_milan.api.server import app, limiter

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"
_LOANS = (SYNTHETIC / "loans.csv").read_bytes()
_COLLATERALS = (SYNTHETIC / "collaterals.csv").read_bytes()
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture(autouse=True)
def _isolated_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh jobs directory and rate-limit counters for every test."""
    monkeypatch.setenv(jobs.JOBS_DIR_ENV, str(tmp_path / "jobs"))
    monkeypatch.delenv(jobs.MAX_UPLOAD_MB_ENV, raising=False)
    limiter.reset()
    return tmp_path / "jobs"


def _files(loans: bytes = _LOANS, collaterals: bytes = _COLLATERALS) -> dict[str, tuple[str, bytes, str]]:
    return {
        "loans": ("loans.csv", loans, "text/csv"),
        "collaterals": ("collaterals.csv", collaterals, "text/csv"),
    }


def _create(client: TestClient, **data: str) -> dict[str, str]:
    response = client.post("/api/jobs", files=_files(), data={"deal_name": "SYNTHETIC_FIXTURE", **data})
    assert response.status_code == 202, response.text
    body: dict[str, str] = response.json()
    return body


# ---------------------------------------------------------------------------
# POST /api/jobs
# ---------------------------------------------------------------------------


def test_create_job_returns_202_with_urls_and_queued_status(_isolated_jobs: Path) -> None:
    client = TestClient(app)
    body = _create(client)
    jid = body["job_id"]
    jobs.validate_job_id(jid)
    assert body == {"job_id": jid, "status_url": f"/api/jobs/{jid}", "result_url": f"/api/jobs/{jid}/result"}
    status = client.get(body["status_url"]).json()
    assert status["state"] == "queued"
    assert status["mode"] == "workbook"
    assert status["deal_name"] == "SYNTHETIC_FIXTURE"
    assert status["stage"] is None and status["error"] is None and status["result"] is None
    assert status["retention_hours"] == 12.0
    # The uploads were streamed into the job directory, byte for byte.
    assert (_isolated_jobs / jid / "inputs" / "loans.csv").read_bytes() == _LOANS
    assert not list(_isolated_jobs.glob("*.tmp"))


def test_create_job_validates_like_process(_isolated_jobs: Path) -> None:
    client = TestClient(app)
    r = client.post("/api/jobs", files=_files(), data={"deal_name": "../x"})
    assert r.status_code == 400 and r.json()["error"] == "invalid_deal_name"
    files = _files()
    files["loans"] = ("loans.csv", _LOANS, "application/octet-stream")
    r = client.post("/api/jobs", files=files, data={"deal_name": "OK"})
    assert r.status_code == 400 and r.json()["error"] == "invalid_content_type"
    r = client.post("/api/jobs", files={"loans": _files()["loans"]}, data={"deal_name": "OK"})
    assert r.status_code == 400 and r.json()["error"] == "missing_field"
    # Nothing was written for any rejected request.
    assert not _isolated_jobs.exists() or not any(_isolated_jobs.iterdir())


def test_create_job_has_no_loan_count_cap_but_honours_upload_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loans file over the synchronous 30,000-row cap is accepted; the
    per-file byte cap comes from ESMA_MILAN_MAX_UPLOAD_MB."""
    client = TestClient(app)
    header, body = _LOANS.split(b"\n", 1)
    big = header + b"\n" + body * 4000  # ~32,000 data rows, ~15 MB
    assert big.count(b"\n") > 30_000
    monkeypatch.setenv(jobs.MAX_UPLOAD_MB_ENV, "20")
    r = client.post("/api/jobs", files=_files(loans=big), data={"deal_name": "BIG"})
    assert r.status_code == 202, r.text
    monkeypatch.setenv(jobs.MAX_UPLOAD_MB_ENV, "1")
    r = client.post("/api/jobs", files=_files(loans=big), data={"deal_name": "BIG"})
    assert r.status_code == 413
    assert r.json()["error"] == "file_too_large"
    assert r.json()["details"]["limit_bytes"] == 1024 * 1024
    # The synchronous endpoint keeps its fixed 50 MiB cap and its row cap.
    r = client.post("/api/process", files=_files(loans=big), data={"deal_name": "BIG"})
    assert r.status_code == 413 and r.json()["error"] == "size_limit_exceeded"
    assert "/api/jobs" in r.json()["message"]


def test_health_reports_upload_cap_and_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok", "max_upload_mb": 50, "job_retention_hours": 12.0}
    monkeypatch.setenv(jobs.MAX_UPLOAD_MB_ENV, "600")
    monkeypatch.setenv(jobs.RETENTION_ENV, "6")
    assert client.get("/api/health").json() == {"status": "ok", "max_upload_mb": 600, "job_retention_hours": 6.0}


# ---------------------------------------------------------------------------
# State machine through GET / result / DELETE
# ---------------------------------------------------------------------------


def test_workbook_job_state_machine(_isolated_jobs: Path) -> None:
    client = TestClient(app)
    body = _create(client)
    jid = body["job_id"]

    r = client.get(body["result_url"])
    assert r.status_code == 409 and r.json()["error"] == "job_not_ready"
    assert r.json()["details"]["state"] == "queued"
    r = client.delete(f"/api/jobs/{jid}")
    assert r.status_code == 409 and r.json()["error"] == "job_not_ready"

    status = worker.process_one(_isolated_jobs, "test-worker")
    assert status is not None and status.state == "done"

    status_body = client.get(body["status_url"]).json()
    assert status_body["state"] == "done"
    assert status_body["stage"] == {"index": 11, "total": 11, "label": "Writing workbook"}
    assert status_body["result"]["filename"] == "2024-06-30 SYNTHETIC_FIXTURE Flattened loans and collaterals.xlsx"
    assert status_body["started_at"] and status_body["finished_at"]

    r = client.get(body["result_url"])
    assert r.status_code == 200
    assert r.headers["content-type"] == _XLSX
    assert r.headers["content-disposition"] == (
        'attachment; filename="2024-06-30 SYNTHETIC_FIXTURE Flattened loans and collaterals.xlsx"'
    )
    wb = openpyxl.load_workbook(io.BytesIO(r.content), read_only=True)
    assert len(wb.sheetnames) == 10

    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert client.get(body["status_url"]).status_code == 404
    assert client.get(body["result_url"]).status_code == 404
    assert client.delete(f"/api/jobs/{jid}").status_code == 404
    assert not (_isolated_jobs / jid).exists()


def test_analysis_job_result_is_analysis_response_json(_isolated_jobs: Path) -> None:
    client = TestClient(app)
    body = _create(client, analysis_only="true")
    assert client.get(body["status_url"]).json()["mode"] == "analysis"
    worker.process_one(_isolated_jobs, "w")
    r = client.get(body["result_url"])
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    data = r.json()
    assert set(data) == {"deal_name", "summary", "stratifications", "execution_summary"}
    assert data["summary"]["loan_count"] == 8
    assert len(data["stratifications"]) == 11


def test_failed_job_result_returns_409_with_job_error(_isolated_jobs: Path) -> None:
    client = TestClient(app)
    r = client.post(
        "/api/jobs",
        files=_files(loans=b"\x00\x01\x02 not,a,csv \xff\xfe"),
        data={"deal_name": "GARBAGE"},
    )
    assert r.status_code == 202
    jid = r.json()["job_id"]
    status = worker.process_one(_isolated_jobs, "w")
    assert status is not None and status.state == "failed"
    body = client.get(f"/api/jobs/{jid}").json()
    assert body["state"] == "failed"
    assert body["error"]["error"] == "invalid_csv"
    assert body["error"]["message"] == "The uploaded data could not be processed."
    r = client.get(f"/api/jobs/{jid}/result")
    assert r.status_code == 409
    assert r.json()["error"] == "invalid_csv"
    # A failed job can be deleted.
    assert client.delete(f"/api/jobs/{jid}").status_code == 204


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["..%2F..%2Fetc%2Fpasswd", "ZZZZ", "0123456789abcdef0123456789abcde", "0123456789ABCDEF0123456789ABCDEF"],
)
def test_malformed_ids_are_404_without_filesystem_access(
    bad_id: str, monkeypatch: pytest.MonkeyPatch, _isolated_jobs: Path
) -> None:
    listed: list[str] = []
    real_is_dir = Path.is_dir

    def spy(self: Path) -> bool:
        listed.append(str(self))
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", spy)
    client = TestClient(app)
    for method, url in (("GET", f"/api/jobs/{bad_id}"), ("GET", f"/api/jobs/{bad_id}/result"), ("DELETE", f"/api/jobs/{bad_id}")):
        r = client.request(method, url)
        assert r.status_code == 404, (method, url, r.text)
        # An id with encoded slashes is decoded into a path that matches no
        # route, so it is FastAPI's own 404; anything that does reach the
        # handler is answered with the API's `job_not_found` shape.
        if "error" in r.json():
            assert r.json()["error"] == "job_not_found"
    assert listed == []


def test_unknown_well_formed_id_is_404() -> None:
    client = TestClient(app)
    jid = jobs.new_job_id()
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert client.get(f"/api/jobs/{jid}/result").status_code == 404
    assert client.delete(f"/api/jobs/{jid}").status_code == 404


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


def test_status_endpoint_rate_limit_applies(_isolated_jobs: Path) -> None:
    """RATE_LIMIT_JOBS defaults to 120/minute: the 121st status poll from
    one client is a 429 in the standard ErrorResponse shape."""
    client = TestClient(app)
    body = _create(client)
    for _ in range(120):
        assert client.get(body["status_url"]).status_code == 200
    r = client.get(body["status_url"])
    assert r.status_code == 429
    assert r.json()["error"] == "rate_limit_exceeded"


def test_create_job_uses_process_rate_limit(_isolated_jobs: Path) -> None:
    client = TestClient(app)
    for _ in range(10):
        _create(client)
    r = client.post("/api/jobs", files=_files(), data={"deal_name": "X"})
    assert r.status_code == 429
