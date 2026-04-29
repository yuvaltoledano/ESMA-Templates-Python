"""Generic data-frame validators used across the pipeline.

`validate_required_columns()` mirrors the R helper at
r_reference/R/utils.R:137-148 - asserts that every name in
`required_cols` is present in the DataFrame, raising a single error
with the full missing list if any are absent.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def validate_required_columns(
    df: pl.DataFrame,
    required_cols: Iterable[str],
    df_name: str,
) -> None:
    """Raise ValueError if any name in `required_cols` is missing from `df.columns`.

    Mirrors r_reference/R/utils.R::validate_required_columns:

        validate_required_columns <- function(df, required_cols, df_name) {
          missing_cols <- setdiff(required_cols, names(df))
          if (length(missing_cols) > 0) {
            stop(glue::glue(
              "ERROR: {df_name} is missing required columns: ",
              "{paste(missing_cols, collapse = ', ')}"
            ))
          }
          invisible(TRUE)
        }
    """
    present = set(df.columns)
    missing = [c for c in required_cols if c not in present]
    if missing:
        raise ValueError(
            f"ERROR: {df_name} is missing required columns: {', '.join(missing)}"
        )
