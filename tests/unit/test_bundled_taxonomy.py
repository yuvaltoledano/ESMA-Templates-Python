"""The ESMA RRE taxonomy bundled with the package.

The taxonomy is a fixed ESMA artefact shipped as package data
(``esma_milan/data/esma_rre_taxonomy.xlsx``) and located through
`read_taxonomy.bundled_taxonomy_path`. Three contracts:

1. It loads through both loaders and carries the 107 RRE fields
   (84 RREL loan fields, 23 RREC collateral fields).
2. It resolves to an existing file whatever the current working
   directory is; the pipeline must not depend on being run from the
   repository root.
3. It is byte-identical to the R reference's copy at
   ``r_reference/inputs/ESMA template taxonomy.xlsx``. The R reference is
   the parity contract, so the two files must move in lockstep: an ESMA
   republication is a submodule bump plus a copy. The check skips when
   the submodule is not checked out (CI's docker job, a plain clone) and
   runs where it is (CI's test job checks the submodule out for it).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from esma_milan.io_layer.read_taxonomy import (
    BUNDLED_TAXONOMY_FILENAME,
    bundled_taxonomy_path,
    load_taxonomy,
    load_taxonomy_formats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
R_REFERENCE_TAXONOMY = REPO_ROOT / "r_reference" / "inputs" / "ESMA template taxonomy.xlsx"

EXPECTED_FIELD_COUNT = 107
EXPECTED_RREL_COUNT = 84
EXPECTED_RREC_COUNT = 23


def test_bundled_taxonomy_loads_with_both_loaders_and_has_107_rre_fields() -> None:
    path = bundled_taxonomy_path()
    names = load_taxonomy(path)
    formats = load_taxonomy_formats(path)

    assert len(names) == EXPECTED_FIELD_COUNT
    assert set(names) == set(formats), "every field has a FIELD NAME and a FORMAT"
    assert sum(code.startswith("RREL") for code in names) == EXPECTED_RREL_COUNT
    assert sum(code.startswith("RREC") for code in names) == EXPECTED_RREC_COUNT
    assert all(code.startswith(("RREL", "RREC")) for code in names)
    # Two anchors the pipeline relies on from the first stage.
    assert names["RREL30"] == "Current Principal Balance"
    assert formats["RREL30"] == "{MONETARY}"


def test_bundled_taxonomy_path_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert not (Path.cwd() / "r_reference").exists(), "the cwd must hold no copy of its own"

    path = bundled_taxonomy_path()

    assert path.is_absolute()
    assert path.is_file()
    assert path.name == BUNDLED_TAXONOMY_FILENAME
    assert path.parent.name == "data"
    assert path.parent.parent.name == "esma_milan"
    # And it is usable from there, not merely present.
    assert len(load_taxonomy(path)) == EXPECTED_FIELD_COUNT


def test_bundled_taxonomy_is_byte_identical_to_the_r_reference_copy() -> None:
    if not R_REFERENCE_TAXONOMY.is_file():
        pytest.skip(
            "r_reference submodule not checked out "
            f"({R_REFERENCE_TAXONOMY.relative_to(REPO_ROOT)} missing); "
            "run `git submodule update --init` to enforce the lockstep check"
        )
    bundled = bundled_taxonomy_path().read_bytes()
    reference = R_REFERENCE_TAXONOMY.read_bytes()
    assert hashlib.md5(bundled).hexdigest() == hashlib.md5(reference).hexdigest()
    assert bundled == reference, (
        "src/esma_milan/data/esma_rre_taxonomy.xlsx differs from the R reference's "
        "ESMA template taxonomy.xlsx; copy the R file over it (cp, not a spreadsheet "
        "save) or bump the submodule, so the two stay in lockstep"
    )
