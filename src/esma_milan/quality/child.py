"""The file part of the audit, and the writing of the report files, each
in a process of its own.

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

The report files are written the same way, after the conversion, for the
same reason turned round: writing millions of detail rows needs working
memory of its own, and in the conversion's process it would be added to
what the conversion still holds.

    parent: write_report(...)     pickles the assessment and the facts of the
                                  run (`assess.RunFacts`: decisions and
                                  counts, no frame of the pool) into the
                                  working directory and starts the child
                                  with `--write-report`
    child:  main()                completes the report, writes the four
                                  files (`report_files.write_report_files`)
                                  and leaves the path of the summary and
                                  the files it could not write in
                                  `report-result.json`

Always a fresh interpreter, never a fork: a process that has imported
Polars must not be forked.

The result file is a pickle of the `FileAssessment` (outcomes, profiles,
inventories, counts; no frame of rows), and what is handed to the second
child is a pickle of that assessment and the `RunFacts`. Each is written by this
package and read back by the same package within one run, from a working
directory this run created (`store.ResultStore.create`). Neither must ever
be read from a directory another party could have written to.

The child prints nothing and logs nothing. Its standard error goes to a
file in the working directory, so a child that dies before it can report
(an import error, a kill) still leaves something to read. That text and
the fault file can quote cells of the user's files; they go into the
summary beside the workbook and are never logged.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from esma_milan.quality.assess import (
    FileAssessment,
    RunFacts,
    assess_files,
    complete_report_from,
)
from esma_milan.quality.report_files import report_filenames, write_report_files
from esma_milan.quality.store import ResultStore

RESULT_FILE = "assessment.pickle"
FAULT_FILE = "assessment-fault.txt"
STDERR_FILE = "assessment-stderr.txt"
STDERR_LINES_KEPT = 20

WRITE_REPORT = "--write-report"
REPORT_FILE = "report.pickle"
REPORT_RESULT_FILE = "report-result.json"
REPORT_FAULT_FILE = "report-fault.txt"
REPORT_STDERR_FILE = "report-stderr.txt"


class AssessmentProcessError(RuntimeError):
    """The assessment process did not finish. `exit_status` is the child's
    exit status (negative for a signal). The message holds what the child
    left behind and belongs in the summary, not in a log event."""

    what = "assessment"

    def __init__(self, exit_status: int, detail: str) -> None:
        self.exit_status = exit_status
        text = f"the {self.what} process ended with exit status {exit_status}"
        super().__init__(f"{text}: {detail}" if detail else text)


class ReportProcessError(AssessmentProcessError):
    """The process that writes the report files did not finish."""

    what = "report-writing"


def _what_the_child_left(
    root: Path, fault_file: str = FAULT_FILE, stderr_file: str = STDERR_FILE
) -> str:
    fault = root / fault_file
    if fault.is_file():
        return fault.read_text(encoding="utf-8", errors="replace").strip()
    stderr = root / stderr_file
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


def _report_command(root: Path, folder: Path, deal_name: str, cutoff: str | None) -> list[str]:
    return [
        sys.executable, "-m", "esma_milan.quality.child", WRITE_REPORT,
        str(root), str(folder), deal_name, cutoff or "",
    ]


def write_report(
    assessment: FileAssessment,
    facts: RunFacts,
    options: Mapping[str, object],
    store: ResultStore,
    folder: Path,
    deal_name: str,
    cutoff: str | None,
) -> tuple[Path, list[tuple[str, str]]]:
    """Complete the report and write its four files into `folder`, in a
    fresh interpreter. The parent hands over the assessment and the facts
    of the run; it loads nothing of the detail itself.
    Returns the summary's path and, for each file that could not be
    written, the step and the type of the exception (the summary has the
    message).

    Raises ReportProcessError if the child exits with a non-zero status or
    without its result. If the wait is interrupted (Ctrl-C), the child is
    ended, the files it had started or finished are removed (an
    interrupted run writes no report) and the interruption propagates."""
    root = store.root
    began = time.time()
    (root / REPORT_FILE).write_bytes(
        pickle.dumps((assessment, facts, dict(options)), protocol=pickle.HIGHEST_PROTOCOL)
    )
    command = _report_command(root, folder, deal_name, cutoff)
    try:
        with (root / REPORT_STDERR_FILE).open("wb") as stderr:
            completed = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr, check=False
            )
    except BaseException:
        for name in report_filenames(deal_name, cutoff).values():
            with contextlib.suppress(OSError):
                (folder / (name + ".part")).unlink(missing_ok=True)
                (folder / (name + ".tmp")).unlink(missing_ok=True)
                written = folder / name
                # Only what this run wrote: a file of an earlier run stays.
                if written.is_file() and written.stat().st_mtime >= began - 1:
                    written.unlink()
        raise
    result = root / REPORT_RESULT_FILE
    if completed.returncode != 0 or not result.is_file():
        raise ReportProcessError(
            completed.returncode, _what_the_child_left(root, REPORT_FAULT_FILE, REPORT_STDERR_FILE)
        )
    written = json.loads(result.read_text(encoding="utf-8"))
    return Path(written["summary"]), [(step, kind) for step, kind in written["faults"]]


def _write_report_main(argv: list[str]) -> int:
    """`--write-report <working directory> <folder> <deal> <cut-off>`."""
    root, folder, deal_name, cutoff = argv
    try:
        # Read only from the working directory this run created: the parent
        # of this process wrote the file a moment ago.
        assessment, facts, options = pickle.loads((Path(root) / REPORT_FILE).read_bytes())
        if not isinstance(assessment, FileAssessment) or not isinstance(facts, RunFacts):
            raise TypeError("the report file does not hold an assessment and the facts of a run")
        report = complete_report_from(assessment, facts, options)
        faults: list[tuple[str, str]] = []
        summary = write_report_files(
            report, Path(folder), deal_name, cutoff or None,
            on_fault=lambda step, exc: faults.append((step, type(exc).__name__)),
        )
        temporary = Path(root) / (REPORT_RESULT_FILE + ".tmp")
        temporary.write_text(json.dumps({"summary": str(summary), "faults": faults}), encoding="utf-8")
        os.replace(temporary, Path(root) / REPORT_RESULT_FILE)
    except Exception as exc:
        with contextlib.suppress(OSError):
            (Path(root) / REPORT_FAULT_FILE).write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8"
            )
        return 1
    return 0


def main(argv: list[str]) -> int:
    """The child's entry point: `python -m esma_milan.quality.child
    <loans> <collaterals> <working directory> <start time>`, or
    `--write-report ...` (`_write_report_main`)."""
    if argv and argv[0] == WRITE_REPORT:
        return _write_report_main(argv[1:])
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
