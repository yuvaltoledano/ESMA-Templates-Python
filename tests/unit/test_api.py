"""Unit tests for the ESMA-MILAN HTTP API (`esma_milan.api`).

These exercise the response *contract* - the status codes, headers, and
body shapes the API promises clients - through FastAPI's `TestClient`,
so no server process is needed. Pipeline internals are covered by the
stage and parity tests; here the focus is the HTTP seam.

Commit 1 (baseline) locks the Day 1 surface: the health check, the
workbook and dry-run success paths, and the known 4xx failure modes.
Commit 2 (hardening) adds rate limiting, error-shape consistency, and
input-validation tests on top.
"""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl
from fastapi.testclient import TestClient

from esma_milan.api.server import app
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"

_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Read the synthetic fixture once - the tests send these bytes as
# multipart uploads. Bytes (not open handles) keep each request
# self-contained with no file-handle lifecycle to manage.
_LOANS_BYTES = (SYNTHETIC / "loans.csv").read_bytes()
_COLLATERALS_BYTES = (SYNTHETIC / "collaterals.csv").read_bytes()
_TAXONOMY_BYTES = (SYNTHETIC / "taxonomy.xlsx").read_bytes()


def _process_files(
    *,
    loans: bool = True,
    collaterals: bool = True,
    taxonomy: bool = False,
    loans_content_type: str = "text/csv",
) -> dict[str, tuple[str, bytes, str]]:
    """Build the multipart ``files=`` mapping for a /api/process request.

    Flags drop a field (to exercise missing-field paths) or, for
    ``loans_content_type``, send a deliberately wrong Content-Type.
    """
    files: dict[str, tuple[str, bytes, str]] = {}
    if loans:
        files["loans"] = ("loans.csv", _LOANS_BYTES, loans_content_type)
    if collaterals:
        files["collaterals"] = ("collaterals.csv", _COLLATERALS_BYTES, "text/csv")
    if taxonomy:
        files["taxonomy"] = ("taxonomy.xlsx", _TAXONOMY_BYTES, _XLSX_MEDIA_TYPE)
    return files


def _oversized_loans_csv(n_rows: int) -> bytes:
    """A loans CSV with ``n_rows`` data rows, for the size-guardrail test.

    The guardrail counts newlines *before* any parse, so the rows need
    not be valid loan records - a real header line plus ``n_rows`` filler
    lines is enough to drive the >30,000-row rejection path without
    committing a giant fixture file.
    """
    header = _LOANS_BYTES.split(b"\n", 1)[0]
    return header + b"\n" + b"x\n" * n_rows


# --- Day 1 regression lock (PR #16) --------------------------------------


def test_dry_run_group_count_matches_stage4_n_groups() -> None:
    """`DryRunResponse.group_count` is Stage 4's connected-component
    count, not the property-node count.

    Regression lock for the bug where `group_count` was populated from
    `collateral_groups.height` (one row per property node) instead of
    the count of distinct `collateral_group_id`s. On the synthetic
    fixture that is 5 groups across 8 property nodes - the two numbers
    differ, so a reintroduced bug fails this test.
    """
    loans = SYNTHETIC / "loans.csv"
    collaterals = SYNTHETIC / "collaterals.csv"
    taxonomy = SYNTHETIC / "taxonomy.xlsx"

    # Stage 4's own count of connected components (collateral groups),
    # taken straight from the pipeline's dry-run output.
    pipeline_result = run_pipeline(
        loans_file_path=loans,
        collaterals_file_path=collaterals,
        taxonomy_file_path=taxonomy,
        deal_name="SYNTHETIC_FIXTURE",
        dry_run=True,
        verbose=False,
    )
    assert pipeline_result.stage4 is not None
    stage4 = pipeline_result.stage4
    stage4_n_groups = stage4.collateral_groups["collateral_group_id"].n_unique()
    assert stage4_n_groups == 5  # 5 components across 8 property nodes

    client = TestClient(app)
    with (
        loans.open("rb") as lf,
        collaterals.open("rb") as cf,
        taxonomy.open("rb") as tf,
    ):
        response = client.post(
            "/api/process",
            files={
                "loans": ("loans.csv", lf, "text/csv"),
                "collaterals": ("collaterals.csv", cf, "text/csv"),
                "taxonomy": ("taxonomy.xlsx", tf, _XLSX_MEDIA_TYPE),
            },
            data={"deal_name": "SYNTHETIC_FIXTURE", "dry_run": "true"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["group_count"] == stage4_n_groups
    # The bug returned the property-node count; pin that it no longer does.
    assert body["group_count"] != stage4.collateral_groups.height


# --- Commit 1: baseline - health + success paths -------------------------


def test_health_endpoint_returns_ok() -> None:
    """GET /api/health returns 200 with the {"status": "ok"} body."""
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_synthetic_fixture_returns_workbook() -> None:
    """POST /api/process (synthetic loans+collaterals, default taxonomy)
    returns a 200 .xlsx download.

    Pins the success contract: the .xlsx media type, an attachment
    Content-Disposition naming the file with the pool cutoff date, and a
    body openpyxl reads as the full 10-sheet workbook.
    """
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "SYNTHETIC_FIXTURE"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == _XLSX_MEDIA_TYPE
    assert response.headers["content-disposition"] == (
        'attachment; filename="2024-06-30 SYNTHETIC_FIXTURE '
        'Flattened loans and collaterals.xlsx"'
    )
    workbook = openpyxl.load_workbook(io.BytesIO(response.content))
    assert len(workbook.sheetnames) == 10


def test_process_dry_run_returns_json() -> None:
    """POST /api/process with dry_run=true returns a 200 JSON
    DryRunResponse carrying the synthetic fixture's known counts and
    resolved aggregation method - no workbook composed.
    """
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "SYNTHETIC_FIXTURE", "dry_run": "true"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "deal_name": "SYNTHETIC_FIXTURE",
        "chosen_aggregation": "by_loan",
        "loan_count": 8,
        "property_count": 12,
        "group_count": 5,
        "warnings": [],
    }


# --- Commit 1: baseline - known 4xx failure modes ------------------------


def test_process_missing_loans_field_returns_400() -> None:
    """POST /api/process without the `loans` upload returns 400 in the
    ErrorResponse shape, not FastAPI's default verbose 422."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(loans=False),
        data={"deal_name": "SYNTHETIC_FIXTURE"},
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"] == "missing_field"
    assert isinstance(body["message"], str) and body["message"]


def test_process_missing_collaterals_field_returns_400() -> None:
    """POST /api/process without the `collaterals` upload returns 400 in
    the ErrorResponse shape."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(collaterals=False),
        data={"deal_name": "SYNTHETIC_FIXTURE"},
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"] == "missing_field"
    assert isinstance(body["message"], str) and body["message"]


def test_process_missing_deal_name_returns_400() -> None:
    """POST /api/process without the `deal_name` form field returns 400
    in the ErrorResponse shape."""
    client = TestClient(app)
    response = client.post("/api/process", files=_process_files(), data={})

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"] == "missing_field"
    assert isinstance(body["message"], str) and body["message"]


def test_process_size_guardrail_returns_413() -> None:
    """A loans upload with >30,000 data rows trips the synchronous-size
    guardrail: 413 with an ErrorResponse naming the limit."""
    client = TestClient(app)
    files = {
        "loans": ("loans.csv", _oversized_loans_csv(30_001), "text/csv"),
        "collaterals": ("collaterals.csv", _COLLATERALS_BYTES, "text/csv"),
    }
    response = client.post(
        "/api/process", files=files, data={"deal_name": "BIG_POOL"}
    )

    assert response.status_code == 413, response.text
    body = response.json()
    assert body["error"] == "size_limit_exceeded"
    assert body["details"]["limit"] == 30_000
    assert "30,000" in body["message"]


def test_process_malformed_csv_returns_400_not_500() -> None:
    """Garbage bytes in the `loans` upload fail inside the pipeline and
    surface as a 4xx client error, never a 500 with leaked internals."""
    client = TestClient(app)
    files = {
        "loans": ("loans.csv", b"\x00\x01\x02 not,a,csv \xff\xfe", "text/csv"),
        "collaterals": ("collaterals.csv", _COLLATERALS_BYTES, "text/csv"),
    }
    response = client.post(
        "/api/process", files=files, data={"deal_name": "GARBAGE"}
    )

    assert 400 <= response.status_code < 500, response.text
    body = response.json()
    assert isinstance(body["error"], str) and body["error"]
    assert isinstance(body["message"], str) and body["message"]


# --- Commit 1: baseline - optional-argument pass-through -----------------


def test_process_filename_in_content_disposition_uses_cutoff_date() -> None:
    """The download filename's date prefix is the loans pool_cutoff_date,
    not a hardcoded or request-time date."""
    header, first_row = _LOANS_BYTES.decode().splitlines()[:2]
    cutoff_idx = header.split(",").index("pool_cutoff_date")
    expected_cutoff = first_row.split(",")[cutoff_idx]

    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "SYNTHETIC_FIXTURE"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"].startswith(
        f'attachment; filename="{expected_cutoff} '
    )


def test_process_with_explicit_taxonomy_argument() -> None:
    """Supplying `taxonomy` as a third upload works the same as relying
    on the server's bundled default taxonomy."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(taxonomy=True),
        data={"deal_name": "SYNTHETIC_FIXTURE", "dry_run": "true"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["loan_count"] == 8


def test_process_with_min_coverage_override() -> None:
    """An explicit in-range `min_coverage` form field is accepted and the
    synthetic pool still processes."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={
            "deal_name": "SYNTHETIC_FIXTURE",
            "dry_run": "true",
            "min_coverage": "0.5",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["loan_count"] == 8


def test_process_with_aggregation_override() -> None:
    """An explicit `aggregation` form field overrides auto-detection;
    requesting by_loan resolves to by_loan in the dry-run summary."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={
            "deal_name": "SYNTHETIC_FIXTURE",
            "dry_run": "true",
            "aggregation": "by_loan",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["chosen_aggregation"] == "by_loan"
