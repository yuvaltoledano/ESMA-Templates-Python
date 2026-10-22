# Decisions

Why the code is the way it is. One entry per decision that someone would
otherwise re-argue; newest at the bottom. Add an entry when a PR changes one
of these or makes a new one. Keep "Where things stand" current.

## Where things stand (updated 2026-10-22)

The port is complete and verified cell for cell against the R reference on
both fixtures. It runs as a CLI, as a synchronous API for pools up to
~30,000 loans, and as background jobs for larger pools through the web page.
It is deployed on one 4 GB Hetzner server (Helsinki) behind Caddy with HTTPS
and a single password, with a 2 GB swap file as a memory safety net. A
300,000-loan pool is expected to take about 16 minutes and 2–3 GB; routine
use at that size calls for the 8 GB tier. Open work is tracked as GitHub
issues labelled `backlog`. Claude Code sessions follow `CLAUDE.md`. The
analyst-facing description of the mapping is
`frontend/public/mapping-reference.html` (version 1.0), served by the web
UI; it is maintained by hand alongside any mapping change.

## Entries

### 2026-05 — The R reference is the contract
The pinned `r_reference/` submodule defines correct output. Parity is
cell-for-cell against R workbooks staged under `tests/fixtures/`, with an
IEEE-754 tolerance (`rel_tol=1e-13`) on six MILAN columns where Polars and R
accumulate in different orders. Deviations from R are allowed only where R
is silently wrong; each is documented at the site in code.

### 2026-10-13 — Nightly parity jobs removed
The Layer-2 live-R job and the Layer-3 fuzzing job were deleted (#23).
Layer-1 parity against the staged workbooks, run in CI and locally, is the
only automated parity gate. Regenerating fixtures requires an R environment
and a re-tagged `r_reference`; see `docs/parity-protocol.md`.

### 2026-10-15 — Input types come from the taxonomy, not from a sample
Polars inferred column types from the first 10,000 rows, which crashed on a
late decimal; readr's equivalent silently nulls a column that is ND for its
first 10,000 rows. Every field the ESMA taxonomy marks `{MONETARY}`,
`{PERCENTAGE}` or `{INTEGER-n}` is now read as Float64; a non-numeric value
in such a field halts the run naming file, field, row and value (#25). Two
deliberate deviations from R follow: late-populated numeric fields are read
correctly, and bad numerics fail loudly instead of becoming missing.
`{Y/N}` and `{LIST}` fields keep whole-file inference, and date fields are
parsed by Stage 1's own parser, so Sheets 6/7 keep their R-native cell
types.

### 2026-10-15 — Non-UTF-8 CSVs are repaired per field on input
Polars requires valid UTF-8; R repairs text at write time. A file that is
not valid UTF-8 is parsed with the `csv` module and every field goes
through the ported `sanitize_utf8` cascade (UTF-8 → Windows-1252 → Latin-1
→ strip) before Polars reads it (#25). Valid files take the old path
unchanged.

### 2026-10-16 — Deployment: one VPS, Docker Compose, Caddy, basic auth
Chosen over PaaS (data would transit third-party storage), Cloudflare
Access (an extra moving part for one user) and Tailscale (not a public
page). Caddy provides HTTPS and the password; the app container publishes
no port; rate limiting keys on `X-Forwarded-For` only because the proxy is
the sole route in (#26). Hostname and password hash live in `.env` on the
server only. Finland (`hel1`) was accepted over Germany; both are EU.

### 2026-10-16 — Server size: 4 GB plus swap now, 8 GB when needed
CPX22 (2 vCPU, 4 GB) was chosen after the 8 GB tier proved expensive. A 2 GB
swap file turns an out-of-memory kill into a slow finish. Measured scaling
is linear at ~3.1 s and ~6.2 MB per thousand loans (#27).

### 2026-10-16 — Background jobs on disk; no broker, no retry, 12-hour retention
Large pools cannot run inside an HTTP request. The queue is a directory on a
shared Docker volume with atomic status files and marker-rename claiming; a
separate `worker` container runs one job at a time (#27). Rejected: Redis
or Celery (infrastructure for one user), in-process jobs (uvicorn workers
share no memory). A job interrupted by a worker restart is failed, not
retried: an out-of-memory kill is deterministic and a retry wastes the run
time. Finished outputs are deleted after 12 hours (`ESMA_MILAN_JOB_RETENTION_HOURS`);
inputs are deleted the moment a job finishes. The page submits every pool
as a job; the synchronous endpoint remains for the CLI-style use and small
pools.

### 2026-10-16 — The writer stays openpyxl for now
xlsxwriter measured ~1.6× faster on this workbook (2.96 s → 1.80 s on the
real fixture — measured by the reviewer on 2026-10-16, not reproduced in
the repository), not the 3–5× hoped for. Write time is linear in cells, so a
300,000-loan workbook is about 13 minutes with openpyxl and about 8 with
xlsxwriter: minutes either way, which is why background jobs were the lever
and the writer was not. Revisit if measured job times on real pools justify
the swap.

### 2026-10-21 — The ESMA taxonomy is bundled; no override
The taxonomy is a fixed ESMA artefact that changes only when ESMA
republishes the template. The `--taxonomy` flag and the upload field were
removed; the file ships as package data (`src/esma_milan/data/`) and a test
keeps it byte-identical to the pinned R reference's copy (#29). If ESMA
republishes: re-tag the R repository, re-pin the submodule, update the
bundled copy, and the test confirms lockstep.

### 2026-10-21 — Frontend: Dashboard design, component structure
The page was restructured into `lib/`, `hooks/` and `components/` with
design tokens in one file and fonts bundled (no third-party requests from a
password-protected page) (#30). Behaviour was preserved item by item from a
38-point checklist.

### 2026-10-21 — Real fixture counterparty names remain unredacted (open)
`tests/fixtures/real_anonymised/` has perturbed identifiers but real lender
and originator names, LEIs and countries, and the EPC provider name
(`tests/fixtures/README.md`). Redacting them is possible without R (they are
pass-through strings) but is a fixture change and has not been decided.

### 2026-10-22 — Working rules for Claude Code; backlog in GitHub issues
Every task runs as a brief from the reviewer, a Phase 1 plan from Claude
Code with no edits, the owner's approval, then one PR squash-merged by the
owner. The standing rules live in `CLAUDE.md` so briefs need not repeat
them; deny rules in `.claude/settings.json` enforce the two that must never
slip (`r_reference/` and `tests/fixtures/`). This file is the durable record
of why, since squash merges drop the commit-level history. Open work moved
from conversation notes to GitHub issues labelled `backlog`.

### 2026-10-22 — The mapping reference is a hand-written page served by the web UI
The analyst-facing documentation moved from the R repository's HTML guides
to `frontend/public/mapping-reference.html`, written from the Python
pipeline as it behaves and with no reference to R, since the reader is a
user of this tool. It is one self-contained HTML file (inline CSS and
script, nothing fetched from other hosts, consistent with the frontend's
no-third-party rule) placed in Vite's `public/` directory so it ships in
the web image and is linked from the top bar without a build step, a
route or a React dependency. It is hand-written rather than generated
from code: the reader order (pool steps, then columns, then summary, then
checklist) is not the code order, and the worked examples are prose. The
cost is that it can drift; the guard is the rule in `CLAUDE.md` that a
mapping change updates the page and bumps its version line, and the
reviewer checks that on every PR that touches the mapping. The synthetic
fixture is the running example for now; a real-pool example is a later
decision.
