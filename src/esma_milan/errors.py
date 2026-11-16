"""Typed error for faults in the uploaded data, and the catalogue of the
places a fault in the data can surface.

This module mirrors nothing in the R reference: R signals every failure
with `stop()`. It exists so a caller of `run_pipeline` can tell a fault
in the data from a bug, and can read which records or values were
involved without parsing a message.

`PipelineDataError` is a `ValueError`. Every site that raises it raised a
plain `ValueError` with the same message before this class existed, so
anything that catches `ValueError` behaves as it did.

`EXPECTED_FAULT_SITES` lists three kinds of site:

  typed             raises `PipelineDataError` with a reason code;
  untyped           a Polars exception (or, at two sites, a built-in
                    `TypeError`) whose cause is the data. These are
                    recorded only. Wrapping them would change the message
                    a caller sees, so they are left as they are;
  not_a_data_fault  a `ValueError` that reads like a data fault but that no
                    input can reach, with the reason.

Sites are named by function, not by line number. An untyped exception is
identified by its site and its meaning, never by its Python type alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

ReasonCode = Literal[
    "duplicate_columns",
    "no_data_rows",
    "non_numeric_value",
    "missing_required_columns",
    "missing_pool_cutoff_date",
    "no_properties_after_filter",
    "no_usable_loan_id_column",
    "loan_id_coverage_below_minimum",
    "missing_identifier_column",
    "no_loans_after_intersection",
    "loan_identifier_mismatch",
]
"""The closed list of reasons a `PipelineDataError` can carry."""

REASON_CODES: frozenset[str] = frozenset(get_args(ReasonCode))

FaultStage = Literal["stage1", "stage2", "stage3", "runner"]


class PipelineDataError(ValueError):
    """A fault in the uploaded data.

    `stage` is where it was detected and `reason` is one of `REASON_CODES`.
    `items` is the complete list of identifiers or values involved, where
    the site has them (an empty tuple where it does not); the message may
    show only a sample of them. `context` holds the scalars the site had
    in hand. What `items` and `context` hold, per reason:

      duplicate_columns               items: the duplicated column names.
                                      context: file.
      no_data_rows                    context: file.
      non_numeric_value               items: (record index, raw text) for
                                      every bad row; the record index is
                                      0-based, so the message's "data row"
                                      is index + 1.
                                      context: file, field_code, column,
                                      n_bad.
      missing_required_columns        items: the missing column names.
                                      context: table.
      missing_pool_cutoff_date        nothing.
      no_properties_after_filter      context: n_loans, n_loans_kept,
                                      n_collateral, loan_reason_counts,
                                      collateral_reason_counts.
      no_usable_loan_id_column        items: (column, value) for every
                                      duplicated value.
                                      context: all_missing (per column),
                                      loan_id_selection.
      loan_id_coverage_below_minimum  items: every unmatched identifier of
                                      the chosen column, sorted.
                                      context: column, matched, total,
                                      coverage, min_coverage,
                                      loan_id_selection.
      missing_identifier_column       items: the missing column name.
                                      context: output_column.
      no_loans_after_intersection     context: column.
      loan_identifier_mismatch        items: ("missing_in_properties" or
                                      "missing_in_loans", identifier) for
                                      every one.
                                      context: n_missing_in_properties,
                                      n_missing_in_loans.

    `loan_id_selection` is the `pipeline.identifiers.LoanIdSelection`
    computed up to the point of failure.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: FaultStage,
        reason: ReasonCode,
        items: tuple[object, ...] = (),
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage: FaultStage = stage
        self.reason: ReasonCode = reason
        self.items: tuple[object, ...] = items
        self.context: dict[str, object] = dict(context or {})


FaultKind = Literal["typed", "untyped", "not_a_data_fault"]


@dataclass(frozen=True)
class FaultSite:
    """One place a fault in the data can surface (or, for
    `not_a_data_fault`, one place that only looks like it)."""

    where: str
    """`<module>:<function>`, importable. For a function that has a private
    twin returning extra data, the public name is given."""

    kind: FaultKind

    exception: str
    """The exception class raised there."""

    reason: ReasonCode | None
    """The reason code; set for `typed` sites only."""

    trigger: str
    """The input that causes it, or why no input can."""

    meaning: str
    """What it means, in plain words."""


_READ = "esma_milan.io_layer.read_csv:read_and_clean"
_CAST = "esma_milan.io_layer.read_csv:_cast_numeric_columns"
_VALIDATE = "esma_milan.pipeline.validate:validate_required_columns"
_RUNNER = "esma_milan.runner:run_pipeline"
_STAGE2 = "esma_milan.pipeline.filters:run_stage2"
_SELECT = "esma_milan.pipeline.identifiers:select_calc_loan_id"
_GENERATE = "esma_milan.pipeline.identifiers:generate_id_column"
_STAGE3 = "esma_milan.pipeline.identifiers:run_stage3"
_FLATTEN = "esma_milan.pipeline.flatten:flatten_loan_collateral"
_NOT_CAUGHT = (
    " Not a ValueError or a Polars error, so callers that treat those as "
    "input errors do not catch it as one today."
)

EXPECTED_FAULT_SITES: tuple[FaultSite, ...] = (
    # --- typed -----------------------------------------------------------
    FaultSite(
        _READ, "typed", "PipelineDataError", "duplicate_columns",
        "two headers with the same cleaned name",
        "a field appears twice in the file",
    ),
    FaultSite(
        _READ, "typed", "PipelineDataError", "no_data_rows",
        "a header and no records",
        "the file has no data",
    ),
    FaultSite(
        _CAST, "typed", "PipelineDataError", "non_numeric_value",
        "text that is neither a number nor an ND token in a numeric field",
        "a value cannot be read as a number",
    ),
    FaultSite(
        _VALIDATE, "typed", "PipelineDataError", "missing_required_columns",
        "a required field is absent",
        "a field the pipeline cannot run without is missing",
    ),
    FaultSite(
        _RUNNER, "typed", "PipelineDataError", "missing_pool_cutoff_date",
        "no readable cut-off date on any loan",
        "the pool has no cut-off date",
    ),
    FaultSite(
        _STAGE2, "typed", "PipelineDataError", "no_properties_after_filter",
        "every collateral row fails a Stage 2 condition",
        "no collateral row is usable",
    ),
    FaultSite(
        _SELECT, "typed", "PipelineDataError", "no_usable_loan_id_column",
        "both loan identifier fields are duplicated or empty",
        "no field identifies a loan",
    ),
    FaultSite(
        _SELECT, "typed", "PipelineDataError", "loan_id_coverage_below_minimum",
        "too few loans have a collateral row",
        "the two files do not match",
    ),
    FaultSite(
        _GENERATE, "typed", "PipelineDataError", "missing_identifier_column",
        "the original obligor or original collateral identifier field is absent",
        "a field Stage 3 needs is missing and the required-column check does "
        "not ask for it",
    ),
    FaultSite(
        _STAGE3, "typed", "PipelineDataError", "no_loans_after_intersection",
        "minimum coverage 0 and no loan matches a collateral row",
        "no loan has a collateral row",
    ),
    FaultSite(
        _STAGE3, "typed", "PipelineDataError", "loan_identifier_mismatch",
        "a collateral row references a loan under the identifier field that "
        "was not chosen, or a loan dropped for a missing identifier",
        "a collateral row has no loan in the pool (the R reference drops "
        "the row and continues)",
    ),
    # --- untyped: recorded, not wrapped ----------------------------------
    FaultSite(
        _READ, "untyped", "polars.exceptions.NoDataError", None,
        "a zero-byte file",
        "the file is empty",
    ),
    FaultSite(
        _READ, "untyped", "polars.exceptions.ComputeError", None,
        "a row with more fields than the header; binary content; an "
        "unterminated quote",
        "the file is not a well-formed rectangular CSV",
    ),
    FaultSite(
        _STAGE2, "untyped", "polars.exceptions.InvalidOperationError", None,
        "account_status holds only numbers",
        "the status field does not hold ESMA codes",
    ),
    FaultSite(
        _STAGE2, "untyped", "polars.exceptions.ColumnNotFoundError", None,
        "no original_valuation_amount field",
        "a field Stage 2 uses is absent and the required-column check does "
        "not ask for it",
    ),
    FaultSite(
        _STAGE2, "untyped", "polars.exceptions.InvalidOperationError", None,
        "collateral_type holds only numbers, or the collateral file's loan "
        "identifier was read as integers",
        "the field does not hold text",
    ),
    FaultSite(
        _SELECT, "untyped", "TypeError", None,
        "both loan identifier fields read as integers (headers that are not "
        "ESMA codes, numeric-looking values)",
        "identifiers were read as numbers." + _NOT_CAUGHT,
    ),
    FaultSite(
        _GENERATE, "untyped", "TypeError", None,
        "obligor or collateral identifier fields read as integers",
        "identifiers were read as numbers." + _NOT_CAUGHT,
    ),
    FaultSite(
        "esma_milan.pipeline.valuation:_add_selection_columns", "untyped",
        "polars.exceptions.ColumnNotFoundError", None,
        "no current_valuation_method field (InvalidOperationError when it "
        "holds numbers)",
        "a field the valuation choice uses is absent or not text",
    ),
    FaultSite(
        "esma_milan.pipeline.valuation:select_valuation_fields", "untyped",
        "polars.exceptions.ColumnNotFoundError", None,
        "no current_valuation_date, original_valuation_method or "
        "original_valuation_date field",
        "a field the valuation choice uses is absent",
    ),
    FaultSite(
        "esma_milan.pipeline.flatten:select_main_property", "untyped",
        "polars.exceptions.ColumnNotFoundError", None,
        "no occupancy_type field (ComputeError when it holds numbers)",
        "the main-property ranking has no occupancy",
    ),
    FaultSite(
        "esma_milan.pipeline.flatten:apply_derived_fields", "untyped",
        "polars.exceptions.ColumnNotFoundError", None,
        "no original_principal_balance field",
        "the original LTV has no numerator",
    ),
    FaultSite(
        _FLATTEN, "untyped", "polars.exceptions.DuplicateError", None,
        "both files carry a field x and the loans file also carries x_right",
        "an input field name collides with a join suffix",
    ),
    FaultSite(
        "esma_milan.pipeline.enriched:compose_loans_enriched", "untyped",
        "polars.exceptions.ComputeError", None,
        "the loans file carries its own numeric field named structure_type",
        "an input field has the name of a pipeline column",
    ),
    # --- not a data fault -------------------------------------------------
    FaultSite(
        _STAGE3, "not_a_data_fault", "ValueError", None,
        "'No properties remaining after filtering.' Not reachable from "
        "data, because a loan that survives the intersection has its "
        "identifier in the collateral set, so its collateral rows survive "
        "too.",
        "a broken invariant, not an input fault",
    ),
    FaultSite(
        _FLATTEN, "not_a_data_fault", "ValueError", None,
        "'collateral groups are unclassified'. Not reachable from data, "
        "because every group is built from at least one edge, so it has at "
        "least one loan and one collateral, and the five structure types "
        "cover every such case.",
        "a broken invariant, not an input fault",
    ),
)
