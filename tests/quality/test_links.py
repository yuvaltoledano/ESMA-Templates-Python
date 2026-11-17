"""The links between collateral records and loan records (specification 5.2)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from esma_milan.quality import assess_files
from esma_milan.quality.links import (
    LIFECYCLES_DIFFER,
    NO_LOAN_LINKED,
    build_links,
    collateral_link_context,
    compare_with_stage3,
    link_counts,
)
from esma_milan.runner import run_pipeline
from tests.quality.builders import Pair

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _links(loans: list[tuple[str, str]], collateral: list[str]) -> list[tuple[int, int, str, str]]:
    left = pl.DataFrame(
        {"record": range(1, len(loans) + 1), "RREL2": [a for a, _ in loans], "RREL3": [b for _, b in loans]},
        schema_overrides={"record": pl.UInt32},
    )
    right = pl.DataFrame(
        {"record": range(1, len(collateral) + 1), "RREC2": collateral}, schema_overrides={"record": pl.UInt32}
    )
    return build_links(left, right).rows()


def test_the_prescribed_and_the_original_convention_are_kept_apart() -> None:
    loans = [("OLD1", "NEW1"), ("OLD2", "NEW2"), ("SAME", "SAME")]
    assert _links(loans, ["NEW1", "OLD2", "SAME", "NOBODY"]) == [
        (1, 1, "new", "raw"),
        (2, 2, "original", "raw"),
        (3, 3, "new", "raw"),
        (3, 3, "original", "raw"),
    ]  # record 4 matches no loan; every loan has a record


def test_a_loan_with_no_collateral_record_and_a_record_with_no_loan() -> None:
    links = _links([("A", "A"), ("B", "B")], ["A", "X"])
    assert links == [(1, 1, "new", "raw"), (1, 1, "original", "raw")]
    frame = pl.DataFrame(links, schema=["collateral_record", "loan_record", "convention", "match"], orient="row")
    counts = link_counts(frame, n_loans=2, n_collateral=2)
    assert counts["collateral_records_matching_no_loan_record"] == 1
    assert counts["loan_records_with_no_collateral_record"] == 1


def test_the_collision_of_issue_45_is_visible_in_the_links() -> None:
    # Loan 1's original identifier is loan 2's new identifier.
    loans = [("X", "N1"), ("O2", "X")]
    assert _links(loans, ["X"]) == [(1, 1, "original", "raw"), (1, 2, "new", "raw")]


def test_padded_identifiers_link_only_after_trimming() -> None:
    assert _links([("L1", "L1")], ["L1 ", " L1", "L1"]) == [
        (1, 1, "new", "trimmed"), (1, 1, "original", "trimmed"),
        (2, 1, "new", "trimmed"), (2, 1, "original", "trimmed"),
        (3, 1, "new", "raw"), (3, 1, "original", "raw"),
    ]
    assert _links([("L1 ", "L1 ")], ["L1 "]) == [(1, 1, "new", "raw"), (1, 1, "original", "raw")]


def test_empty_cells_and_nd_tokens_link_nothing() -> None:
    loans = [("", ""), ("ND5", "ND5"), (" ", "ND1")]
    assert _links(loans, ["", "ND5", " ", "ND1"]) == []


def test_link_context_lifecycle_and_retained() -> None:
    loans = pl.DataFrame(
        {
            "record": [1, 2, 3, 4],
            "RREL2": ["A", "B", "C", "D"],
            "RREL3": ["A", "B", "C", "D2"],
            "lifecycle": ["reporting-active (presumed)", "final-inactive (presumed)",
                          "reporting-active (presumed)", "not established"],
            "population": ["Active, retained", "Inactive", "Active status, zero balance", "Eligibility unknown"],
        },
        schema_overrides={"record": pl.UInt32},
    )
    collateral = pl.DataFrame(
        {"record": [1, 2, 3, 4, 5], "RREC2": ["A", "B", "D", "NOBODY", "A"]},
        schema_overrides={"record": pl.UInt32},
    )
    context = collateral_link_context(build_links(loans, collateral), loans, 5)
    assert context["linked_lifecycle"].to_list() == [
        "reporting-active (presumed)", "final-inactive (presumed)",
        "not established",  # linked under the original convention only
        NO_LOAN_LINKED, "reporting-active (presumed)",
    ]
    assert context["linked_retained"].to_list() == [True, False, False, False, True]
    assert context["links_new"].to_list() == [1, 1, 0, 0, 1]
    assert context["links_original"].to_list() == [1, 1, 1, 0, 1]


def test_a_record_of_two_loans_whose_lifecycles_differ() -> None:
    loans = pl.DataFrame(
        {"record": [1, 2], "RREL2": ["A1", "A2"], "RREL3": ["A", "A"],
         "lifecycle": ["reporting-active (presumed)", "final-inactive (presumed)"],
         "population": ["Active, retained", "Inactive"]},
        schema_overrides={"record": pl.UInt32},
    )
    collateral = pl.DataFrame({"record": [1], "RREC2": ["A"]}, schema_overrides={"record": pl.UInt32})
    context = collateral_link_context(build_links(loans, collateral), loans, 1)
    assert context["linked_lifecycle"].to_list() == [LIFECYCLES_DIFFER]
    assert context["linked_retained"].to_list() == [True]


@pytest.mark.parametrize(
    ("fixture", "selected", "new", "original"),
    [("synthetic", "RREL3", (12, 8), (0, 0)), ("real_anonymised", "RREL2", (1952, 1295), (1952, 1295))],
)
def test_agreement_with_stage3_on_the_fixtures(
    fixture: str, selected: str, new: tuple[int, int], original: tuple[int, int]
) -> None:
    folder = FIXTURES / fixture
    assessment = assess_files(folder / "loans.csv", folder / "collaterals.csv")
    result = run_pipeline(loans_file_path=folder / "loans.csv", collaterals_file_path=folder / "collaterals.csv",
                          deal_name="T", dry_run=True, verbose=False)
    assert assessment.links is not None and result.stage3 is not None
    loans = assessment.records("loans")
    assert loans is not None
    comparison = compare_with_stage3(loans, assessment.links, result.stage2, result.stage3.decisions)
    assert comparison.evaluated and comparison.agrees and comparison.selected == selected
    assert comparison.missing_identifier == () and comparison.without_collateral == ()
    raw = assessment.links.filter(pl.col("match") == "raw")
    for convention, (n_collateral, n_loans) in (("new", new), ("original", original)):
        side = raw.filter(pl.col("convention") == convention)
        assert (side["collateral_record"].n_unique(), side["loan_record"].n_unique()) == (n_collateral, n_loans)
    assert assessment.links.filter(pl.col("match") == "trimmed").height == 0


def _lossy_pair() -> Pair:
    pair = Pair()
    loan, collateral = pair.loans[0], pair.collateral[0]
    pair.loans = [{**loan, "RREL2": f"L{i}", "RREL3": f"L{i}", "RREL5": f"B{i}", "RREL4": f"B{i}"} for i in (1, 2, 3)]
    pair.collateral = [
        {**collateral, "RREC2": "L1", "RREC4": "P1", "RREC3": "P1"},
        {**collateral, "RREC2": "L2", "RREC4": "P2", "RREC3": "P2", "RREC5": "GUAR"},
    ]  # L2's only record is excluded by Stage 2; L3 has no record at all
    return pair


@pytest.mark.filterwarnings("ignore:select_calc_loan_id:UserWarning")
def test_the_loans_stage3_drops_are_the_loans_the_links_say_it_drops(tmp_path: Path) -> None:
    loans_path, collateral_path = _lossy_pair().write(tmp_path)
    assessment = assess_files(loans_path, collateral_path)
    result = run_pipeline(loans_file_path=loans_path, collaterals_file_path=collateral_path, deal_name="T",
                          dry_run=True, verbose=False, min_coverage=0.0)
    loans = assessment.records("loans")
    assert loans is not None and assessment.links is not None and result.stage3 is not None
    decisions = result.stage3.decisions
    assert decisions is not None
    comparison = compare_with_stage3(loans, assessment.links, result.stage2, decisions)
    assert comparison.agrees
    assert comparison.without_collateral == (2, 3) and comparison.missing_identifier == ()
    # A decision that disagrees with the links is reported by record.
    tampered = dataclasses.replace(decisions, loans_without_collateral=decisions.loans_without_collateral.head(1))
    differing = compare_with_stage3(loans, assessment.links, result.stage2, tampered)
    assert differing.evaluated and not differing.agrees and differing.differing_records == (3,)
    none = compare_with_stage3(loans, assessment.links, None, None)
    assert not none.evaluated and none.reason == "the pipeline did not return a Stage 3 result"
