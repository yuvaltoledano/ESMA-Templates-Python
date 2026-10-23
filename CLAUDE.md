# Working in this repository

Standing rules for every Claude Code session on ESMA-Templates-Python. The owner
sets direction and merges; a reviewer drafts task briefs and reviews PRs; you
implement. Task briefs assume these rules and do not repeat them. Where a brief
conflicts with this file, the brief wins for that task; say in the plan that
you noticed the conflict.

## What this is

A Python port of an R pipeline that turns ESMA Annex 2 RRE loan-level
templates into Moody's MILAN input (`README.md`). The R reference under
`r_reference/` (a git submodule pinned to a tag) is the contract: the Python
output must match it cell for cell on both fixtures. The service runs on one
server with Docker Compose and Caddy (`docs/deploy.md`); large pools run as
background jobs (`docs/api.md`). Decisions that shaped the code and why are in
`docs/decisions.md`; read it before proposing a change to anything it covers.

## Before changing anything

1. **Read the entire repository first**: every source file, test, config,
   workflow, script and document, not only the files a task names. This is
   how you catch the second place a change must land and the test that pins
   current behaviour. Skip these unless the task touches them: fixture data
   under `tests/fixtures/` (CSVs and R workbooks; read
   `tests/fixtures/README.md`), lockfiles (`uv.lock`,
   `frontend/package-lock.json`), the bundled taxonomy under
   `src/esma_milan/data/`, `node_modules/`, build output, and `r_reference/`
   beyond the files that mirrored modules cite.
2. **Plan, then wait.** Every task runs in two phases. Phase 1: edit nothing
   except the plan file (below); verify the brief's claims against the code
   with file:line, write the plan, and stop. Phase 2 starts only when the
   owner says the plan is approved. Expect revision rounds between the two.
3. If a brief's claim is wrong, say so and say what it changes. Do not work
   around it silently.

A task may arrive as a full brief or as a GitHub issue. The same two phases
apply either way.

## The plan file

- Write the Phase 1 plan to `.plans/<branch-name>.md` (gitignored). It is the
  review surface: the owner and the reviewer read it whole and annotate it.
  Contents: verified claims with file:line (and any that are wrong), the
  approach, files to change with reasons, code snippets for the non-obvious
  changes, tests, risks, anything you would do differently, and a todo list
  of checkbox items grouped by commit.
- If the brief asks for a research round, first write
  `.plans/<branch-name>-research.md` (how the relevant code works, what
  depends on it, what pins its behaviour), stop, and write the plan only
  after the owner confirms the research.
- **Revision rounds.** The owner returns the file with inline notes, each a
  blockquote beginning `NOTE:`. Address every note, update the plan, remove
  the note, and add one line per note to a "Revisions" section at the end
  (what was asked, what changed). Do not implement. Repeat until approved.
- **Phase 2.** Work through the todo list in order and tick each item in the
  plan file as it is done, so the file always shows where things stand. If
  the approved plan proves wrong mid-way, stop and say so rather than
  improvising. If the owner says the changes were reverted, start again
  from the narrowed scope; do not try to recover the discarded approach.
- If the owner cannot open files in your environment (a cloud session),
  include the full plan file in your Phase 1 message as well.

## Never

- Modify anything under `r_reference/` or move the submodule pointer.
- Modify anything under `tests/fixtures/` (inputs, R workbooks, README)
  unless the brief says so explicitly for a named file.
- Modify or delete `REPO_ASSESSMENT.md` if it is in the working tree; it is
  intentionally uncommitted.
- Add a dependency, change `uv.lock`, or add infrastructure (databases,
  brokers, services) without the brief calling for it.
- Put a hostname, password, hash, IP address, key, or any personal detail in
  the repository. Deployment values live in `.env` on the server;
  `.env.example` carries placeholders.
- Widen scope. Anything you notice outside the brief goes under "Observed,
  not touched" in the PR, not into the diff.
- Reword comments or docs that are accurate. Fix what is wrong or missing.

Deny rules in `.claude/settings.json` back the first two items. That file is
committed and shared; do not work around a rule by moving or overriding it
in the git-ignored `.claude/settings.local.json`. A brief that needs a
fixture change (including `tests/fixtures/README.md`) will say to adjust
the rule in the same PR; never change the rules otherwise.

## The parity gate

Any change under `src/` (including "no behaviour change" refactors) must show
the workbooks are unchanged: build both fixtures' workbooks from `origin/main`
(a git worktree is the simplest way) and from the branch, and diff them with
`scripts/run_parity_check.py` (expect PASS on both), in addition to
`uv run pytest`, which runs the Layer-1 parity tests against the R workbooks.
Report the verbatim output. A deliberate deviation from the R reference is
allowed only when the brief asks for it; it is documented at the site in code
and named in the PR.

## Conventions

- Python: `uv`, 3.12, Polars; `uv run ruff check src tests scripts`,
  `uv run mypy` (strict), `uv run pytest` must all be clean. Modules that
  mirror R code cite the R file and line range (`Mirrors r_reference/R/<file>:<lines>`).
- Frontend: React + Vite + Tailwind; `cd frontend && npm run lint && npm run build`.
  Tokens in `frontend/src/lib/tokens.js`; network only in `lib/api.js` and
  the hooks; components take props. No run-time requests to third-party hosts.
- Errors are loud: a bad input halts with a message naming file, field, row
  and value rather than silently nulling (see `docs/decisions.md`).
- Docs are part of the change: `README.md`, `docs/api.md`, `docs/deploy.md`,
  `docs/parity-protocol.md`, `frontend/README.md`, `docs/analysis.md`,
  `docs/decisions.md` and this file must stay true after the PR. When a PR
  makes or changes a decision, add an entry to `docs/decisions.md` and
  update its "Where things stand" section. A PR that changes the mapping
  (filters, identifier choice, valuation selection, grouping, allocation,
  a MILAN column rule, a code map or an Execution Summary metric) also
  updates `frontend/public/mapping-reference.html` and bumps its version
  line.

## Branches, commits, PRs

- One task, one branch, one PR, squash-merged by the owner. Start the branch
  from the current `origin/main`. If your session's branch name was used by
  a merged PR, reset it from `origin/main` and say so.
- A few logically separate commits, each individually green, so the history
  reads well before the squash.
- Use `.github/PULL_REQUEST_TEMPLATE.md`. The description's "What and why"
  section carries the verified-claims checklist; the template's other
  sections carry the file list with reasons, behaviour changes, R
  deviations, verbatim verification output, and "Observed, not touched".
- Report what did not run in your sandbox (Docker daemon, image pulls) and
  what covers it instead (the CI `docker` job, the owner's server test).
- The owner tests PR branches on the server before merging when a change
  touches deployment, the frontend, the API or the job path. Write
  `docs/deploy.md` changes so that test is a copy-paste sequence.

## When a brief is silent

Choose the quieter option: the smaller diff, the existing pattern, the
standard library, no new abstraction. Say what you chose and why in the plan.
