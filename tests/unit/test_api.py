"""Unit tests for the ESMA-MILAN HTTP API (`esma_milan.api`).

Day 1 covers the synchronous `POST /api/process` dry-run path. The
focus here is the response *contract* - the values the API promises
clients - rather than pipeline internals, which the stage tests cover.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from esma_milan.api.server import app
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"

_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


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
