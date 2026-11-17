"""Data quality report on the two uploaded files.

Mirrors nothing in the R reference. The rules are specified in
docs/data-quality/specification.md.

The report is additive: the pipeline's output, its success and its failure
are exactly as they are without it. The audit reads the two files itself
and no audit frame ever reaches a pipeline stage.

  assess_files      read and assess the two files (before the pipeline)
  complete_report   complete the report from the pipeline's result or
                    from the exception it raised
  write_report_files  write the four report files beside the workbook:
                    the detail CSV, the dropped-loans CSV, the report
                    workbook and, last, the summary
  write_summary     write the summary file alone
"""

from esma_milan.quality.assess import (
    FileAssessment,
    QualityReport,
    assess_files,
    complete_report,
)
from esma_milan.quality.export import summary_dict, summary_filename, write_summary
from esma_milan.quality.report_files import report_filenames, write_report_files

__all__ = [
    "FileAssessment",
    "QualityReport",
    "assess_files",
    "complete_report",
    "report_filenames",
    "summary_dict",
    "summary_filename",
    "write_report_files",
    "write_summary",
]
