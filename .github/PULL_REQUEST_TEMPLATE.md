## What and why

## Behaviour change
<!-- "None", or a list. A workbook cell that can change on any input is a behaviour change. -->

## Deviations from the R reference
<!-- "None", or: what R does, what this does, why. Must also be documented in code. -->

## Files changed
<!-- One line per file: path — reason. -->

## Verification (verbatim output)
- [ ] Read the entire repository before changing anything
- [ ] `uv run ruff check src tests scripts`
- [ ] `uv run mypy`
- [ ] `uv run pytest` — N passed
- [ ] Before/after parity on both fixtures (`scripts/run_parity_check.py <main output> <branch output>`): PASS / PASS, or "n/a: no src/ change"
- [ ] `git submodule status` — pointer unchanged; `tests/fixtures/` untouched
- [ ] Frontend: `npm run lint && npm run build`, or "n/a"

## Observed, not touched
