"""The file part of the audit in a process of its own.

Mirrors nothing in the R reference.

Reading two large files as text and assessing them uses a few gigabytes,
and the process does not hand that memory back when the work is done: the
allocator keeps the freed pages. If the audit ran in the same process as
the conversion, the conversion would start on top of them. So the command
line starts a fresh interpreter for `assess_files`. It writes the row-level
detail to the store and one small result file, and exits; everything it
used is then returned to the system and the conversion runs in a process
that looks exactly as it does without the report.

    parent: run_assessment(...)   starts `python -m esma_milan.quality.child`
    child:  main()                assess_files(store=...), writes the store and
                                  `assessment.pickle`; on a fault, the fault file

Always a fresh interpreter, never a fork: a process that has imported
Polars must not be forked.

The result file is a pickle of the `FileAssessment` (outcomes, profiles,
inventories, counts; no frame of rows). It is written by this package and
read back by the same package within one run, from a working directory
this run created (`store.ResultStore.create`). It must never be read from
a directory another party could have written to.

The child prints nothing and logs nothing. Its standard error goes to a
file in the working directory, so a child that dies before it can report
(an import error, a kill) still leaves something to read. That text and
the fault file can quote cells of the user's files; they go into the
summary beside the workbook and are never logged.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from esma_milan.quality.assess import FileAssessment, assess_files
from esma_milan.quality.store import ResultStore

RESULT_FILE = "assessment.pickle"
FAULT_FILE = "assessment-fault.txt"
STDERR_FILE = "assessment-stderr.txt"
STDERR_LINES_KEPT = 20


class AssessmentProcessError(RuntimeError):
    """The assessment process did not finish. `exit_status` is the child's
    exit status (negative for a signal). The message holds what the child
    left behind and belongs in the summary, not in a log event."""

    def __init__(self, exit_status: int, detail: str) -> None:
        self.exit_status = exit_status
        text = f"the assessment process ended with exit status {exit_status}"
        super().__init__(f"{text}: {detail}" if detail else text)


def _what_the_child_left(root: Path) -> str:
    fault = root / FAULT_FILE
    if fault.is_file():
        return fault.read_text(encoding="utf-8", errors="replace").strip()
    stderr = root / STDERR_FILE
    if stderr.is_file():
        lines = stderr.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return "\n".join(lines[-STDERR_LINES_KEPT:])
    return ""


def _command(loans: Path, collaterals: Path, root: Path, started_at: datetime) -> list[str]:
    """The child's command line: this interpreter, started afresh."""
    return [
        sys.executable, "-m", "esma_milan.quality.child",
        str(loans), str(collaterals), str(root), started_at.isoformat(),
    ]


def run_assessment(
    loans: Path, collaterals: Path, store: ResultStore, started_at: datetime
) -> FileAssessment:
    """Assess the two files in a fresh interpreter and return what it
    found. The rows are in `store`; the returned object holds none.

    Raises AssessmentProcessError if the child exits with a non-zero
    status or without its result file. Ctrl-C ends the child and
    propagates."""
    command = _command(loans, collaterals, store.root, started_at)
    with (store.root / STDERR_FILE).open("wb") as stderr:
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr, check=False
        )
    result = store.root / RESULT_FILE
    if completed.returncode != 0 or not result.is_file():
        raise AssessmentProcessError(completed.returncode, _what_the_child_left(store.root))
    # Read only from the working directory this run created (see the module
    # docstring): the file was written a moment ago by our own child.
    assessment = pickle.loads(result.read_bytes())
    if not isinstance(assessment, FileAssessment):
        raise AssessmentProcessError(completed.returncode, "the result file is not an assessment")
    return assessment


def main(argv: list[str]) -> int:
    """The child's entry point: `python -m esma_milan.quality.child
    <loans> <collaterals> <working directory> <start time>`."""
    loans, collaterals, root, started = argv
    try:
        store = ResultStore.open(Path(root))
        assessment = assess_files(
            Path(loans), Path(collaterals),
            started_at=datetime.fromisoformat(started), store=store,
        )
        temporary = Path(root) / (RESULT_FILE + ".tmp")
        temporary.write_bytes(pickle.dumps(assessment, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temporary, Path(root) / RESULT_FILE)
    except Exception as exc:
        with contextlib.suppress(OSError):
            (Path(root) / FAULT_FILE).write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8"
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
