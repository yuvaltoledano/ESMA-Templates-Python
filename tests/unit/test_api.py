"""Unit tests for the ESMA-MILAN HTTP API (`esma_milan.api`).

These exercise the response *contract* - the status codes, headers, and
body shapes the API promises clients - through FastAPI's `TestClient`,
so no server process is needed. Pipeline internals are covered by the
stage and parity tests; here the focus is the HTTP seam: the health
check, the workbook and dry-run success paths, the known 4xx failure
modes, rate limiting, error-shape consistency and input validation.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import openpyxl
import pytest
from fastapi.testclient import TestClient

from esma_milan.api.server import app, limiter
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


def _process_files(
    *,
    loans: bool = True,
    collaterals: bool = True,
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


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Give every test a clean per-IP rate-limit counter.

    The Limiter's in-memory store is module-level state on the shared
    `app`, so without this reset the dozens of /api/process requests
    across this file would exhaust the 10/minute limit and later tests
    would see spurious 429s. Function-scoped and autouse: each test -
    including the dedicated rate-limit test, which re-enters the fixture
    like any other - starts counting from zero.
    """
    limiter.reset()


# --- Regression lock (PR #16) --------------------------------------------


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

    # Stage 4's own count of connected components (collateral groups),
    # taken straight from the pipeline's dry-run output.
    pipeline_result = run_pipeline(
        loans_file_path=loans,
        collaterals_file_path=collaterals,
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
    ):
        response = client.post(
            "/api/process",
            files={
                "loans": ("loans.csv", lf, "text/csv"),
                "collaterals": ("collaterals.csv", cf, "text/csv"),
            },
            data={"deal_name": "SYNTHETIC_FIXTURE", "dry_run": "true"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["group_count"] == stage4_n_groups
    # The bug returned the property-node count; pin that it no longer does.
    assert body["group_count"] != stage4.collateral_groups.height


# --- Health + success paths ----------------------------------------------


def test_health_endpoint_returns_ok() -> None:
    """GET /api/health returns 200 with status "ok" plus the two
    deployment settings the page reads at load (upload cap, retention)."""
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "max_upload_mb", "job_retention_hours"}


def test_process_synthetic_fixture_returns_workbook() -> None:
    """POST /api/process (synthetic loans+collaterals) returns a 200
    .xlsx download.

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


# --- Known 4xx failure modes ---------------------------------------------


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


# --- Optional-argument pass-through --------------------------------------


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


# --- Rate limiting -------------------------------------------------------


def test_process_rate_limit_returns_429_on_11th_request() -> None:
    """The 11th /api/process request inside the window returns 429 with
    the rate_limit_exceeded ErrorResponse; the first 10 are admitted."""
    client = TestClient(app)
    for i in range(10):
        admitted = client.post(
            "/api/process",
            files=_process_files(),
            data={"deal_name": "RATE_LIMIT_PROBE", "dry_run": "true"},
        )
        assert admitted.status_code != 429, f"request {i + 1}: {admitted.text}"

    throttled = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "RATE_LIMIT_PROBE", "dry_run": "true"},
    )
    assert throttled.status_code == 429, throttled.text
    body = throttled.json()
    assert body["error"] == "rate_limit_exceeded"
    assert isinstance(body["message"], str) and body["message"]


# --- Error-response consistency ------------------------------------------


def test_validation_error_returns_consistent_error_response() -> None:
    """A present-but-malformed field (min_coverage outside [0.0, 1.0])
    yields a 422 in the ErrorResponse shape, not FastAPI's default
    verbose validation body."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={
            "deal_name": "SYNTHETIC_FIXTURE",
            "dry_run": "true",
            "min_coverage": "2.5",
        },
    )

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"] == "validation_error"
    assert set(body) == {"error", "message", "details"}


def test_internal_error_does_not_leak_stack_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected pipeline failure returns a generic 500: no
    traceback, no project-tree file paths, no pipeline-internal module
    names in the response body. Full detail goes to the server log.
    """
    leaky = RuntimeError(
        "polars panic in /home/user/ESMA-Templates-Python/src/"
        "esma_milan/runner.py\nTraceback (most recent call last): ..."
    )

    def _boom(*args: object, **kwargs: object) -> object:
        raise leaky

    monkeypatch.setattr("esma_milan.api.handlers.run_pipeline", _boom)

    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "SYNTHETIC_FIXTURE"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["error"] == "internal_error"
    for token in ("Traceback", "polars", "runner.py", "/home/", "ESMA-Templates"):
        assert token not in response.text, f"leaked {token!r} to the client"


# --- Input validation ----------------------------------------------------


def test_process_file_too_large_returns_413() -> None:
    """A single upload over the 50 MB per-file limit is rejected with
    413 file_too_large before its bytes are processed."""
    oversized = b"x" * (50 * 1024 * 1024 + 1)
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files={
            "loans": ("loans.csv", oversized, "text/csv"),
            "collaterals": ("collaterals.csv", _COLLATERALS_BYTES, "text/csv"),
        },
        data={"deal_name": "BIG_FILE"},
    )

    assert response.status_code == 413, response.text
    body = response.json()
    assert body["error"] == "file_too_large"
    assert body["details"]["field"] == "loans"


def test_process_wrong_content_type_returns_400() -> None:
    """An upload sent as application/octet-stream instead of a CSV type
    is rejected with 400 invalid_content_type naming the accepted
    types."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(loans_content_type="application/octet-stream"),
        data={"deal_name": "WRONG_TYPE"},
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"] == "invalid_content_type"
    assert body["details"]["field"] == "loans"
    assert "text/csv" in body["message"]


def test_process_deal_name_path_traversal_rejected() -> None:
    """A deal_name with path-traversal characters is rejected with 400
    invalid_deal_name - it must never reach the output filename."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "../../../etc/passwd", "dry_run": "true"},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid_deal_name"


def test_process_deal_name_too_long_rejected() -> None:
    """A deal_name longer than 100 characters is rejected with 400
    invalid_deal_name."""
    client = TestClient(app)
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "A" * 101, "dry_run": "true"},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid_deal_name"


def test_process_deal_name_with_special_chars_rejected() -> None:
    """deal_name values carrying quotes, newlines, or null bytes - the
    Content-Disposition header-injection vectors - are rejected with
    400 invalid_deal_name."""
    client = TestClient(app)
    for bad_name in ('deal"name', "deal\nname", "deal\x00name"):
        response = client.post(
            "/api/process",
            files=_process_files(),
            data={"deal_name": bad_name, "dry_run": "true"},
        )
        assert response.status_code == 400, (bad_name, response.text)
        assert response.json()["error"] == "invalid_deal_name"


# --- Deployment configuration: run() env vars, proxy-aware rate-limit key, log colours


def test_run_passes_env_host_port_workers_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from esma_milan.api import server as server_module

    captured: dict[str, object] = {}

    def fake_run(app_ref: str, **kwargs: object) -> None:
        captured["app"] = app_ref
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setenv("ESMA_MILAN_HOST", "127.0.0.1")
    monkeypatch.setenv("ESMA_MILAN_PORT", "9123")
    monkeypatch.setenv("ESMA_MILAN_WORKERS", "2")

    server_module.run()

    assert captured == {
        "app": "esma_milan.api.server:app",
        "host": "127.0.0.1",
        "port": 9123,
        "workers": 2,
    }


def test_run_defaults_match_previous_hard_coded_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """With none of the variables set, laptop behaviour is unchanged."""
    import uvicorn

    from esma_milan.api import server as server_module

    captured: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app_ref, **kw: captured.update(kw))
    for var in ("ESMA_MILAN_HOST", "ESMA_MILAN_PORT", "ESMA_MILAN_WORKERS"):
        monkeypatch.delenv(var, raising=False)

    server_module.run()

    assert captured == {"host": "0.0.0.0", "port": 8000, "workers": 4}


def _request_with(headers: dict[str, str], client: tuple[str, int] = ("10.0.0.2", 5000)) -> object:
    """A minimal Starlette Request with the given headers and peer address."""
    from starlette.requests import Request

    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/health",
        "headers": raw_headers,
        "client": client,
        "query_string": b"",
    }
    return Request(scope)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("203.0.113.9, 10.0.0.2", "203.0.113.9"),  # client, then proxy: first wins
        ("203.0.113.9", "203.0.113.9"),
        (" 203.0.113.9 ,10.0.0.2", "203.0.113.9"),  # whitespace tolerated
    ],
)
def test_rate_limit_key_uses_first_forwarded_address_when_proxy_trusted(
    monkeypatch: pytest.MonkeyPatch, header: str, expected: str
) -> None:
    from esma_milan.api.server import _rate_limit_key

    monkeypatch.setenv("ESMA_MILAN_TRUST_PROXY", "1")
    request = _request_with({"X-Forwarded-For": header})
    assert _rate_limit_key(request) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("headers", [{}, {"X-Forwarded-For": ""}, {"X-Forwarded-For": " , "}])
def test_rate_limit_key_falls_back_to_peer_without_usable_header(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    from esma_milan.api.server import _rate_limit_key

    monkeypatch.setenv("ESMA_MILAN_TRUST_PROXY", "1")
    request = _request_with(headers)
    assert _rate_limit_key(request) == "10.0.0.2"  # type: ignore[arg-type]


@pytest.mark.parametrize("flag", [None, "0", "true", "yes"])
def test_rate_limit_key_ignores_header_unless_flag_is_exactly_1(
    monkeypatch: pytest.MonkeyPatch, flag: str | None
) -> None:
    """Off by default, and only the literal "1" turns it on: a forged
    header must never influence the key when the app is exposed directly."""
    from esma_milan.api.server import _rate_limit_key

    if flag is None:
        monkeypatch.delenv("ESMA_MILAN_TRUST_PROXY", raising=False)
    else:
        monkeypatch.setenv("ESMA_MILAN_TRUST_PROXY", flag)
    request = _request_with({"X-Forwarded-For": "203.0.113.9"})
    assert _rate_limit_key(request) == "10.0.0.2"  # type: ignore[arg-type]


def test_rate_limit_counts_forwarded_clients_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the app: with the proxy flag on, two forwarded
    clients get separate /api/health budgets; with it off they share the
    peer's. Exhausts the 60/minute health limit, so it relies on the
    autouse limiter reset."""
    monkeypatch.setenv("ESMA_MILAN_TRUST_PROXY", "1")
    client = TestClient(app)
    for _ in range(60):
        assert client.get("/api/health", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 200
    assert client.get("/api/health", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 429
    assert client.get("/api/health", headers={"X-Forwarded-For": "203.0.113.2"}).status_code == 200


def test_console_renderer_has_no_colours_when_stdout_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Container logs are a pipe: no ANSI escape codes may appear."""
    import structlog

    from esma_milan.api import server as server_module

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    server_module._configure_logging()
    structlog.get_logger("colour_probe").info("probe_event", key="value")
    out = capsys.readouterr().out
    assert "probe_event" in out
    assert "\x1b[" not in out
