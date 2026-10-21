"""A background job and the synchronous endpoint must produce the same output.

For the synthetic fixture: the workbook a job writes to disk is
cell-identical (parity.diff.diff_workbooks) to the one POST /api/process
returns in memory, and the analysis JSON served from a job equals the
synchronous AnalysisResponse, key order included. This pins that the
worker path (disk output, progress callback, output.tmp rename) changes
nothing in the pipeline's result.

The order check matters: dict equality ignores key order, so a job's
analysis.json written with sorted keys compared equal to the synchronous
body while listing the stratifications alphabetically. The page renders
tiles in the order the JSON carries, which is the registry order
(esma_milan.analysis.STRATIFICATIONS) that docs/analysis.md documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from esma_milan.analysis import STRATIFICATIONS
from esma_milan.api import jobs, worker
from esma_milan.api.server import app, limiter
from esma_milan.parity import diff_workbooks, format_report

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jobs.JOBS_DIR_ENV, str(tmp_path / "jobs"))
    limiter.reset()


def _files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "loans": ("loans.csv", (SYNTHETIC / "loans.csv").read_bytes(), "text/csv"),
        "collaterals": ("collaterals.csv", (SYNTHETIC / "collaterals.csv").read_bytes(), "text/csv"),
    }


def test_job_workbook_is_cell_identical_to_synchronous_workbook(tmp_path: Path) -> None:
    client = TestClient(app)

    sync = client.post("/api/process", files=_files(), data={"deal_name": "SYNTHETIC_FIXTURE"})
    assert sync.status_code == 200, sync.text
    sync_path = tmp_path / "sync.xlsx"
    sync_path.write_bytes(sync.content)

    created = client.post("/api/jobs", files=_files(), data={"deal_name": "SYNTHETIC_FIXTURE"})
    assert created.status_code == 202, created.text
    status = worker.process_one(jobs.jobs_root(), "w")
    assert status is not None and status.state == "done"
    job_result = client.get(created.json()["result_url"])
    assert job_result.status_code == 200
    assert job_result.headers["content-disposition"] == sync.headers["content-disposition"]
    job_path = tmp_path / "job.xlsx"
    job_path.write_bytes(job_result.content)

    report = diff_workbooks(job_path, sync_path)
    assert report.passed, f"\n{format_report(report)}\n"

    # And both still match the staged R reference.
    against_r = diff_workbooks(job_path, SYNTHETIC / "expected_r_output.xlsx")
    assert against_r.passed, f"\n{format_report(against_r)}\n"


def test_job_analysis_json_equals_synchronous_analysis_response() -> None:
    client = TestClient(app)
    sync = client.post(
        "/api/process", files=_files(), data={"deal_name": "SYNTHETIC_FIXTURE", "analysis_only": "true"}
    )
    assert sync.status_code == 200, sync.text

    created = client.post(
        "/api/jobs", files=_files(), data={"deal_name": "SYNTHETIC_FIXTURE", "analysis_only": "true"}
    )
    assert created.status_code == 202
    status = worker.process_one(jobs.jobs_root(), "w")
    assert status is not None and status.state == "done"
    job_result = client.get(created.json()["result_url"])
    assert job_result.status_code == 200

    job_body = job_result.json()
    sync_body = sync.json()
    assert job_body == sync_body

    # Equality above ignores key order; the page does not. json.loads
    # keeps document order, so these lists are the orders each response
    # carries. Both must be the registry order.
    registry_order = list(STRATIFICATIONS)
    assert list(job_body["stratifications"]) == registry_order
    assert list(sync_body["stratifications"]) == registry_order
    assert list(job_body) == list(sync_body)
