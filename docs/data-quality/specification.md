# Data quality report: specification

This is the agreed specification for the data quality report on uploaded
ESMA Annex 2 RRE files: every column against the taxonomy, consistency
across columns, records and files, and how the result is shown. It is the
planning document as the owner signed it on 16 November 2026, with its
history removed. Each implementation PR has a brief that says which
sections it implements; where a brief and this file differ, the brief
wins for that PR and this file is corrected in the same PR.

How to read it:

- Section numbers are those of the planning document, so they are not
  contiguous. Sections 2 and 13 (history and observations from sample
  pools), the background lists of section 15 and the revision log are
  omitted. A reference to one of them, or to a numbered "question" or
  lettered "assumption", is historical: **section 15.1 is the decision
  record and settles all of them.**
- "I" is the reviewer who wrote the plan. "The package" is the ESMA
  disclosure templates package v1.3.1 (workbook `DRAFT1auth.099.001.04`,
  XSD v1.3.0). `rre-source-mapping.csv`, beside this file, has one row
  per field with its source, type, ND options, published rules and the
  checks that apply.
- The PR split in 11.3 is the original one. The first PR (the parity gate
  and the stage interface) is merged. The brief of each later PR states
  its own scope.
- Nothing here is implemented until the PR that says so. Implemented so
  far: the parity gate and the stage interface (10.5, 9.1); and, in
  `src/esma_milan/quality/`, sections 3.2 to 3.5, 4, 5, 6 and 8.1 as far
  as they need only the two files (the paragraphs headed "As
  implemented" say exactly what), with the summary file of 10.3.

---

## 1. What the report must do

Uploaded pools are expected to be of poor quality. The report answers
three separate questions and keeps them separate, because they can
legitimately differ:

1. **Reporting compliance.** Does each supplied value comply with the
   ESMA Annex 2 requirements that can be verified from the two files:
   field present, value format, ND options, code lists, and the
   conditional rules written into the field definitions?
2. **Processing and acceptance.** Can the pipeline process the files,
   what did it exclude and why, and does the pool meet the analyst's
   acceptance policy for producing a MILAN file?
3. **Plausibility.** Which values or patterns are unusual enough to need
   an originator's explanation, without being a rule breach?

A correctly used ND5 is compliant and still leaves an input unavailable
to MILAN. A lower-case `true` may be processable and still not be the
prescribed representation. The report says both things, in different
places.

It also states a fourth thing: **assessment completeness**. Every check in
the catalogue ends as evaluated (with or without findings), not
applicable, or not evaluated with the reason. A check that could not run
never looks like a pass.

Every record of both files and every physical column is assessed, with
complete affected-row lists (source file, logical record number, raw
identifiers, field, raw value, expected), so the export can be corrected
and uploaded again. The report is produced on every run, including
blocked and rejected ones. The checks never change a value the pipeline
reads or writes: the ten canonical workbook sheets keep passing the
existing parity comparison (section 10.5 states that guarantee
precisely).

## 3. Framework

### 3.1 Three judgments and a completeness status

| Judgment | Values | Decided by |
|---|---|---|
| **Reporting compliance** | two facts shown together: *No violations* or *Violations (n rules, n fields, n records)*; and *fully assessed* or *partly assessed (n checks, n records not evaluable)*. | only results of kind **violation** (3.2) from rules that have a regulatory source, over all records of both files. A sourced rule can also emit an accepted alternative, context, a question or a limitation; none of those is a compliance failure. |
| **MILAN policy outcome** | Eligible / Eligible with warnings / Blocked (Stop) / Rejected by policy / Not determined / Unavailable (the outcome could not be calculated: a technical failure, or an unexplained population difference, G8); plus "override applied". Eligible with warnings permits release exactly as Eligible does. Unavailable is never shown as eligible or as a rejection and cannot be overridden. | Stops, and the four rejection rules applied to critical checks on the retained active pool (section 8). This is permission. Whether a workbook exists is read from the artifact manifest (10.1), never from this value. |
| **Plausibility** | count of observations needing explanation | findings whose basis is "plausibility"; never blocks anything on its own unless the check is on the critical list by the owner's decision (C25) |
| **Assessment completeness** | Complete / Partial (n checks with records not evaluable) / File unreadable | per-check coverage counts (3.4). Questions outside the declared two-file scope (section 4) are listed separately and do not make a run "partial". |

The earlier words Accepted, Accepted with warnings and Rejected remain
the plain-language summary of the policy outcome. An override releases
the MILAN file and leaves the underlying outcome ("Rejected by policy" or
"Not determined") and every finding in place, with "override applied"
beside it; an overridden undetermined outcome stays undetermined and is
not relabelled as a proven rejection or a pass.

### 3.2 What every rule declares

| Attribute | Values |
|---|---|
| `check_id` | stable id (P3c, C12a, E2) |
| `basis` | **E** ESMA reporting requirement; **L** pipeline constraint (what this pipeline can read or needs); **X** mathematical or relational consistency between supplied values; **A** analyst acceptance policy; **P** plausibility or distribution. A rule may carry a `regulatory_source` as well as basis X (for example "origination date not after cut-off" is a comparison and a published validation rule); it then counts in the compliance judgment, once. |
| `source` | for E: article, annex field or XSD type (section 4); for L: file and function; for A and P: the config constant and its reason |
| `severity` | Stop (the pipeline cannot produce MILAN), Warn, Info |
| `applies_to` | the condition under which a record is in scope (all loan records, records with a default status, property collateral rows, ...) |
| `critical` | whether it counts towards rejection rules 2 and 4 |
| `defect_key` | the underlying defect it measures, so two views of one defect count once (8.4) |
| result kind | each row a rule emits is one of: **violation** (a regulatory requirement, a consistency, or a stated processing prerequisite of this pipeline is established as breached; the rule's basis and `regulatory_source` say which), **accepted alternative** (a representation that is not the template form but is valid, such as XML-form booleans), **context** (history, distributions), **question** (evidence suggests a problem but does not establish it), **limitation** (could not be assessed), **policy trigger** (an approved acceptance condition is met; basis A). Severity does not carry this distinction; the result kind does. Only violations from rules with a regulatory source enter the compliance judgment; violations and policy triggers on counting checks enter rejection rules 2 and 4. |
| `needs` | prerequisites: fields that must be present and parseable, the other file, a stage result |

Wording follows the result kind first, then the basis. Only a
**violation** says "does not comply with" (E) or "contradicts" (X); a
question from an E-sourced rule says "may not comply; please confirm",
a limitation says what could not be assessed, context states a fact. P
findings say "unusual; please confirm" and never assert an error. L
findings say what the pipeline does with the value.

Basis is always one of E, L, X, A, P. Basis, severity and result kind
are three separate attributes in the catalogue. Where a table in this
plan shows a result kind in brackets after the basis, such as
"P (context)" or "E (question)", it is the kind that row always emits,
not a fourth basis.

**Acceptance-policy conditions.** Two kinds of row enter the rejection
rules: violations, and rows that meet an approved acceptance condition
(basis A) even though nothing is established as breached. C25 is the
example: RREL31 equal to the loan's own balance is not an ESMA breach or
an impossibility, it has no regulatory source, and it counts towards
rejection, overridably, because the owner approved that condition. Such a
row has result kind **policy trigger** and never appears in the
compliance judgment. Rule 3 is a third thing: it measures observed loss
from the retained pool from the pipeline's exclusions and needs no
record to be labelled a reporting violation.

**Counted processing effects.** E2's processing part, E3 and F3(b)
count towards rules 2 and 4 as violations of a processing
prerequisite. Each is an
observed exclusion of a collateral record by the pipeline. Its result
kind is a **violation of a stated processing prerequisite**, basis L,
with no regulatory source: "this pipeline needs a property type (E2), a
valuation above 0 (E3), a matching identifier (F3(b)) to use the
record". It can therefore never enter the compliance judgment, and the
regulatory question that E2 also raises about ND5 never counts. The
unit is the loan, once, and by the owner's decision (15.1, decision 7) only
a retained loan **left with no usable collateral record** is in the
affected set, that is, a loan the pipeline loses. A loan that survives
through another record does not count towards rules 2 and 4; its
excluded record is still reported as a Warn, with the note that the
loan's MILAN property value and LTV were computed without that record.
With 100 retained loans of which 11 each have one excluded extra
property record and survive through another, E2 lists 11 warnings and
counts 0; rule 3 shows no balance lost. Rules 2 and 4 then measure the
lost loans by number where rule 3 measures them by balance, attributed
to E2, E3 or F3(b) by the waterfall's bookkeeping priority (8.2).

### 3.3 Scope: applicability, retention and counting are three things

- **Applicability** is a property of the rule and the raw record. Every
  rule runs on every record it applies to: active, redeemed, repurchased,
  defaulted, non-property collateral, collateral of inactive loans.
  Status-specific rules keep their condition ("a loan in an active
  status must ...") and are not run on other statuses.
- **Retention** is what the pipeline does with the record (section 8's
  population labels). Each finding row carries the record's population.
- **Counting** is the acceptance policy: only findings on the retained
  active pool feed rules 2 and 4. Findings on other records are fully
  listed and labelled "does not count towards the MILAN decision".

**Regulatory lifecycle is a third thing, and it is not pipeline
retention.** "Inactive" in section 8 only means "outside the pipeline's
four retained status codes". For ESMA an exposure is active while it is
expected to generate cash flows, so a defaulted exposure with recoveries
expected is active for reporting. An exposure that became inactive is
reported one last time, and Q&A 1296 prescribes that final submission:
identifiers, cut-off and account status populated; **current** monetary
and numeric fields reported as 0; every other field where ND5 is
available set to ND5; the remaining fields left as in the previous
submission; the collateral section completed in the same way. So on a
final-inactive exposure a property record can legitimately keep RREC5 =
RBLD with RREC9 = ND5, carry ND5 in RREC7 and RREC16, show 0 in RREC12
and RREC13, and retain an old valuation method.

**The redemption date is an exception to that pattern.** Q&A 1296 refers
to Q&A 1341 for RREL9, which I have now read: an active exposure reports
ND5; a redeemed exposure reports the date the redemption occurred; a
defaulted exposure whose recovery process is completed reports the date
the recovery was completed; every other way of becoming inactive
(repurchase, substitution and the like) reports ND5. So RDMD with a
valid RREL9 not after the cut-off and RREL30 = 0 is exactly the
prescribed final report, and the date is required, not an ND5-eligible
field carrying a value.

**A submitted default status requires the default fields.** Q&A 1324 is
status-based: default amount and date are completed for an exposure
reported in default. That holds for "default, recoveries not
determinable" as much as for a reporting-active default, because the
requirement follows from the status the record itself declares and not
from the lifecycle presumption. Only the final-inactive branch (a
default status with a valid completed-recovery date in RREL9) follows
the Q&A 1296 pattern instead. C43 applies this.

A snapshot of two files cannot prove lifecycle (it depends on expected
future cash flows and on the previous submission), so the label is an
operational **presumption** and is always shown as one. It is derived
only from the lifecycle the record **claims** through its status and
event fields, never from the values that the lifecycle pattern is then
used to check; otherwise a value that breaks the pattern would remove the
record from the check.

| Claimed lifecycle | Evidence used |
|---|---|
| reporting-active (presumed) | status PERF, ARRE, RNAR or RARR |
| final-inactive (presumed) | status RDMD or a repurchase status; or a default status with RREL9 a valid date not after the cut-off (recovery process ended) |
| default, recoveries not determinable | a default status without such a redemption date. A positive balance does not establish that cash flows are still expected. |
| not established | status empty, invalid, or OTHR |

Three things are then reported separately: the claimed lifecycle; the
evidence and any conflict in it (for example RDMD with an invalid or
future RREL9); and whether the remaining values are consistent with the
claim. The last is a check of its own, **L1**, which runs only on
final-inactive (presumed) records and puts every field in exactly one
of four categories:

| Category | Fields | Expected on the final report | L1 result |
|---|---|---|---|
| 1. Always populated, and specific exceptions | identifiers, cut-off date, account status; RREL9 under Q&A 1341 (the redemption or completed-recovery date; ND5 on a repurchase or other inactive case); RREL8 on a repurchase status (the field's own definition: the date the exposure was repurchased) | the value | none from L1. These fields keep their own rules (C27). A valid RREL9 date not after the cut-off on RDMD or on a completed recovery, and a valid RREL8 date on a repurchase status, are never questioned, and a date used as lifecycle evidence is not then tested as "a value where ND5 was expected". |
| 2. Current monetary and numeric fields | the "current" amounts, rates and counts: RREL30, RREL39, RREL43, RREL46, RREL67, RREL68, RREC12, RREC13 (provisional list) | 0 | a value other than 0 is a question citing Q&A 1296 (a redeemed loan with a positive balance). A date is never in this category: RREL66 (Date Last In Arrears) allows ND5 and is in category 3. |
| 3. Other fields where ND5 is available | every remaining field whose ND5 flag is YES, including RREL29 and RREL66 | ND5 | a value is a question citing Q&A 1296 |
| 4. Preserved historical fields | every remaining field (ND5 flag NO, not a current numeric field), for example RREL23, RREC17, RREC18, RREC19 | unchanged from the previous submission | none: any valid value is consistent, and equality with the previous submission is outside the two files |

A first assignment of all 107 fields now exists as a column of the
source mapping file (4.3): 13 fields in category 1, 8 in category 2, 63
in category 3 and 23 in category 4. Categories 3 and 4 follow
mechanically from the ND5 flag. The category 2 list and the RREL8
exception are my reading; the Q&A sweep (4.4) found nothing that settles
either, so they stay qualified in the rules themselves; they
are not settled.

**RREL8, C27 and L1 reconciled.** Without an exception a repurchased
loan could not be reported cleanly: a date would draw an L1 question
and ND5 would draw a C27 finding. I found no Q&A on RREL8 itself. Q&A
1341 settles RREL9 on a repurchase (ND5) and says nothing against the
repurchase date. The resolution, with its precedence: the field's own
definition comes first, so on a repurchase status a valid RREL8 date is
the expected value (category 1, no L1 question); ND5 in RREL8 on a
repurchase status is a C27 **question**, not a violation, because the
general Q&A 1296 instruction can be read to cover it; an empty cell or
an invalid marker is a violation as everywhere. This is to be confirmed
by a later source if one appears; the sweep found none.

Consistent final reports under these rules, each with no lifecycle
finding: redemption (RDMD, RREL9 a date, RREL8 = ND5, RREL30 = 0);
repurchase (REBR, RREL8 a date, RREL9 = ND5, RREL30 = 0); completed
recovery (DFLT, RREL9 the completion date, RREL8 = ND5, RREL30 = 0,
RREL71 and RREL72 = ND5). An L1 question is never
suppressed and never a reason to relabel the record.

Rules whose meaning depends on lifecycle (P3f, P8 on zeros, E2, E5, E8,
E9, E10, E17, C43) state the lifecycle branch in their own row. Lifecycle
uncertainty limits only the conclusion that depends on it: the regulatory
part of such a rule becomes a question or a limitation, while an invalid
format, an ND code not allowed for the field, an empty cell, an observed
loader failure and an observed pipeline exclusion are reported exactly as
for any other record. Current numeric fields and ND5-eligible fields
follow different branches, so there is no single blanket ND5 rule.

One exception, and it is not a business-quality finding: a value the
strict loader cannot parse (A5) blocks processing wherever the record
sits, including an inactive row. It is a global processing blocker with
basis L, reported as such; the inactive record's other findings still do
not count.

### 3.4 Check outcomes

Every check reports four counts, because findings and incomplete
coverage can both be true of one check: **applicable** records,
**evaluated**, **failed** (records with a violation or a policy trigger
only; accepted alternatives, context and questions are counted in their
own columns and are not failures), and **not evaluable**, the last with its
reasons (a prerequisite field absent; a prerequisite value unparseable or
a permitted ND, where that value is an input to a comparison and not the
thing the check tests: an omission rule stays evaluable when its target
is empty; the other file unreadable; a pipeline stage did not
complete; the comparison basis not established). Example: 100 applicable,
80 evaluated, 10 failed, 20 not evaluable because RREL7 is ND. The
one-word label shown beside the counts is derived from them: not
applicable (0 applicable), passed, findings, findings and partly
evaluated, partly evaluated, not evaluated.

Two kinds of "not assessed" are kept apart: a check **blocked within the
declared scope** (it could have run on these two files and did not),
which lowers completeness; and a question **outside the declared scope**
(history, registries, the true reason behind an ND code), which is listed
once as a scope statement and never lowers completeness.

A malformed number makes the calculations that need it unavailable for
that record and nothing else. Checks on other columns, other records and
the other file continue. "Complete report" means every check is accounted
for, not that every calculation ran.

A structurally unreadable CSV (no header, an unterminated quote, not
decodable) is the boundary: the report says the file could not be split
into records, gives the position of the failure, runs nothing that needs
records from that file, and does not invent counts.

**As implemented.** A file is reported as not splittable into records
when: it has no header (an empty file); a record has more fields than
the header (every such record is listed with its record number, the
physical line it starts on and its field count, not only the first); a
quote opens and never closes (the record and line where it opens); or
the two parsers of 5.1 read the header differently, which is what line
ends made of a lone carriage return do. A file with a header and no
record is readable and has 0 records. "Not decodable" cannot occur: the
repair cascade of the loader accepts every byte, and a repaired file is
readable with the repair reported. The other file is still assessed.

**Completeness in the releases before the last.** Every rule of the
catalogue that is not implemented yet is a check that did not run. So
until all families are implemented the report always shows assessment
completeness as *partial*, with the rules not implemented in the release
counted and listed apart from the rules that could not be evaluated on
the files, and compliance coverage as partial for the same reason.

### 3.5 Severity

- **Stop**: the pipeline cannot produce a MILAN file, or a decision of
  the owner's makes the condition blocking (A9). Never overridable. Basis is
  always L or A.
- **Warn**: a compliance breach, a contradiction, or a plausibility
  observation on a field that matters.
- **Info**: context (profiles, distributions, conventions detected).

Compliance breaches in fields the pipeline does not use are Warn in the
compliance judgment and never count towards the MILAN decision.

---

## 4. Sources and the rule catalogue

The bundled taxonomy is the field catalogue, but its FORMAT label and two
ND flags do not define every requirement. Each E rule cites one of:

| Source | Used for | Version recorded on the report |
|---|---|---|
| Delegated Regulation (EU) 2020/1224 | Article 9(3) ND options, Article 11 identifiers, Annex 2 field content and ND allowances | regulation reference |
| Implementing Regulation (EU) 2020/1225, Annex I | the format table: `{ALPHANUM-n}`, `{MONETARY}`, `{PERCENTAGE}`, `{NUTS}`, `{Y/N}` and the rest | regulation reference |
| ESMA disclosure XSD and validation rules package | XML types (decimal digits, `Max4PositiveNumber`, `xs:boolean`), published validation rules | workbook `DRAFT1auth.099.001.04` version 1.3.1 and XSD version 1.3.0, read in full for the RRE performing template on 15 November 2026 |
| ESMA securitisation Q&A | interpretation, e.g. Q&A 1283: ND allowance does not make a field optional | Q&A id |
| ISO 3166-1 alpha-2, ISO 4217, Eurostat NUTS 3 by classification year | membership lists | list date or NUTS year |
| Bundled taxonomy workbook | field names, CONTENT TO REPORT text | file hash |

The catalogue is a reviewed data structure in the package (one entry per
field: format, ND flags, code list, conditional rules, source
references), not something derived at run time. The regex extraction of
LIST codes from CONTENT TO REPORT stays as a **test** that the reviewed
code lists equal what the taxonomy text contains; a regex match alone is
not treated as proof that a field's rules are captured. The application
rule-set gets its own version string, shown with the taxonomy hash and
the ESMA package version on the report's Summary.

The XSD and validation-rule package is not in the repository today.
Transcribing the relevant constraints into the catalogue with a citation
is enough; whether to add the package itself as reference data is
settled in principle (section 15). The published validation rules must be read in
full during the brief's research round; this plan includes the ones the
review named and the ones visible in the field definitions.

### 4.1 Reconciliation still to do (a prerequisite, not a formality)

The catalogue in sections 6 and 7 was written from the Annex 2 field
definitions, the ITS format table and a handful of Q&As. Its
reconciliation with the published validation rules of the package is
complete (4.2, with the per-field evidence in the file named in 4.3).
Its reconciliation with the full set of Q&As is not. Before any PR
brief is written,
each applicable published requirement gets a reviewed row: source and
version; applicability conditions; lifecycle exceptions; how the
requirement is represented in a CSV; the check that implements it and
the result kind it emits; whether it contributes to the compliance
judgment or to the rejection rules; its acceptance case; and any
limitation. This reconciliation remains the main prerequisite for
freezing the specification. The package part of it is now done (4.2);
the Q&A part is not complete. Items known before the package arrived,
all now confirmed in it:

| Source | Requirement | Where it lands |
|---|---|---|
| Validation workbook 1.3.1 (confirmed) | RREL12 year greater than 1980 and less than 2100 | P6 on RREL12 |
| same | RREL59 between 0 and 100 inclusive | P6 on RREL59 |
| same | RREL65 not later than the cut-off | P6 on RREL65 |
| same | RREL23 not later than the cut-off | C3, with a regulatory source |
| same | non-negativity rules attached to monetary amount and sign structures | reconciled field by field in 4.2; P6 |
| Q&A 1296 | final submission of an inactive exposure: current numeric fields 0, other ND5-eligible fields ND5, remaining fields unchanged; collateral section likewise | lifecycle label (3.3); L1, with one disposition per field (the four categories of 3.3) recorded in the catalogue; P3f, P8, E2, E5, E8, E9, E10, E17 |
| Q&A 1341 (read 14 November) | RREL9: ND5 on an active exposure; the redemption date on a redeemed exposure; the date the recovery process was completed on a defaulted exposure; ND5 for every other inactive case (repurchase, substitution) | RREL9, C27; the exception in L1 (category 1); lifecycle evidence (3.3) |
| Q&A 1283 | an ND allowance does not make a field optional; ND5 is not a substitute for unavailable data | P1, P3 |
| Q&A 1313 | when no obligor or collateral uses a NUTS 3 code (all `ZZZ`), any year may be entered in RREL12 | P11 exception |
| Q&A 1322 | DTI and LTV for loan parts use a pro-rated denominator; a second lien uses the property value after deducting the prior lien | C47, E9 |
| Q&A 1323(c) | for a guarantee, RREC8 is the guarantee holder's priority on the guarantor, possibly first lien | E17, RREC8 |
| Q&A 1323(d) | shared collateral is reported once per exposure with the same identifier; RREC13, RREC17, RREC16 and RREC21 are pro-rated | E10, E9 |
| Further Q&As | see 4.4 for the full sweep | 4.4 |
| Q&A 1343 (read 16 November) | RREL77: the full deposit counts whatever a guarantee scheme covers; construction deposits are included; for savings sub-participations the reporter chooses consistently between savings in RREL77 with RREL30 and RREL29 net, or savings out of RREL77 with balances gross; the amount is what the national legal framework allows to be offset (for example capped at arrears) | RREL77, C44, C30 |
| Q&A 1324 | RREL71 and RREL72 are completed for an exposure reported in default, whatever is known about recoveries; they are not changed when a defaulted exposure performs again; they are updated on a further default | C10, C43, RREL71, RREL72 |

The reconciliation covers, per rule: the published validation
requirement and how it is represented in a CSV; Q&A exceptions and
lifecycle conditions; amount and sign treatment; the difference between
a required value, a permitted ND and non-applicability; and the exact
evidence a cross-record comparison needs before it can assert anything.
The remaining securitisation Q&As on Annex 2 fields have to be read the
same way. Where the Annex text and a Q&A differ (RREL71 "enter 0 if not
in default" against Q&A 1324), the rule follows the Q&A's lifecycle
condition and cites both.

**Not verifiable from two current-period files** (listed on the report as
"not assessed, outside the files", never as passes): whether ND1 rather
than ND2 or ND3 is the true reason; whether an ND5 is justified where no
other field establishes applicability; identifier stability across
reporting periods (Article 11(3), RREL2/RREL4 "must not amend"); whether
an LEI belongs to the named entity; whether RREL79 and RREL82 match the
GLEIF name for the LEI (a workbook rule that needs the GLEIF database);
the workbook's history rules (RREL2 never changes; a new RREL3 or RREC4
must not already exist in earlier submissions); whether a NUTS code is the obligor's
true region; registry-level relationships; anything in the other Annex 2
sections or the investor report.

### 4.2 What the ESMA package says (reconciled 15 November 2026)

Read: the workbook's main sheet filtered to the RRE performing template
(every element, validation rule, pattern, multiplicity and code value),
its change log, and the XSD types behind them. ND options and code lists
were compared with the bundled taxonomy by script, field by field. The
non-performing branch of the workbook belongs to the NPE add-on template
and is outside this plan.

| # | Finding in the package | Effect on the plan |
|---|---|---|
| 1 | 104 of the 107 fields have an element. For all 104 the ND options offered (ND1 to ND3, dated ND4, ND5) and every code list are identical to the bundled taxonomy. | The taxonomy's ND flags and code lists are confirmed as the catalogue. No change. |
| 2 | Every field element has a minimum occurrence of 1. RREL65 is the one repeatable element (`[1..n]`); the rest are `[1..1]`. | Confirms P1: an ND allowance does not make a field optional. |
| 3 | RREL18, RREL28 and RREC22 have no element. Currency is an attribute on every amount (`Ccy`, pattern `[A-Z]{3}`, rule "must contain a valid ISO currency code"). So 104 fields have elements and three are carried as attributes of the amounts they describe. | The requirement that every amount has a stated, valid currency is a published requirement. How a CSV states it is this application's layout (6.2, five cases). The three dedicated fields remain template fields in the 107 (P1, C30). |
| 4 | Amounts are `xs:decimal`, 18 digits, 5 fractional, with a magnitude facet of not below 0 and a separate optional sign indicator (true or absent is plus, false is minus). The facet alone therefore does not forbid a negative economic amount; the field-level business rule decides. That rule, "the numerical field must be equal to or greater than 0", is attached (on the amount, or for RREL20 and RREL32 on the sign element) to RREL16, RREL20, RREL29, RREL31, RREL32, RREL33, RREL41, RREL61, RREL64, RREL67, RREL71, RREL77, RREC13, RREC17 and RREC21. It is absent from RREL30, RREL39, RREL73 and RREL74. The change log explains the history: amendment 73 (1.2.0) stated the rule for a list of fields, amendment 74 (1.2.0) removed it for RREL30, RREL39 and RREL74 "to clarify that [they] may be negative", and amendment 79 (1.3.1) deleted it for RREL73. | **Changed.** P6 now carries this exact list. A negative value in the first group is a violation (RREL29 was "not reconciled"). A negative RREL30, RREL39, RREL73 or RREL74 is valid reporting. A leading minus in the CSV is this application's representation of the sign indicator. Workbook rows per field are in the source mapping file. |
| 5 | "Must be equal to or less than the Data Cut-off Date" is attached to eleven dates: RREL7, RREL8, RREL9, RREL23, RREL63, RREL65, RREL66, RREL72, RREC15, RREC19, RREC20. | **Changed.** All eleven get the regulatory source in P6, so a breach is a compliance violation; the plan had the source on RREL23 and RREL65 only. |
| 6 | The workbook has no date-against-cut-off restriction on RREL24, RREL36, RREL51, RREL53, RREL55, RREL60 or RREL62. The 1.3.1 change log (amendment 79) deleted the restrictions on maturity, grace-period end, lock-out end, fee end and interest revision dates because they rejected valid files. | The package does not require these dates to be before or after the cut-off, and the plan cites no workbook rule for them. That is all this finding says. A comparison justified by a field's own definition keeps its basis: C13 (RREL51 is the date the rate **next** changes, so a date on or before the cut-off contradicts the definition; X, sourced to the Annex field text and Q&A 1334, not to the workbook), C3, C17(b), C41, C42. |
| 7 | RREL6: "must be in the past". | **Changed.** A cut-off date after the assessment date is a violation (was plausibility), owned by A10(a) with the boundary defined there. |
| 8 | Percentages are `xs:decimal`, 11 digits, 10 fractional, with no sign or range rule except RREL59 (0 to 100). | Confirms "no general 0 to 100". A negative percentage is schema-valid everywhere, so no published rule covers a negative DTI; C24 treats it as context or a question, never a violation on the sign alone. |
| 9 | `Max4PositiveNumber` (RREL25, RREL47, RREL58, RREL68, RREC8): no fractional digits, 4 digits, not below 0. RREL12 is `xs:gYear`, greater than 1980 and less than 2100. Y/N fields are `xs:boolean`. | Confirmed. No change. |
| 10 | Text fields are `xs:string` with a length of 1 to n only. The XSD has no capital-letter or ASCII restriction. RREL1 and RREC1 have the pattern LEI, `N`, four digits, two digits. | **Clarified.** P7(a) length is in the XSD and the ITS. P7(b) capitals and ASCII rests on the ITS format table alone and the XSD accepts lower case; the finding says so and still never counts. |
| 11 | LEI: pattern `[A-Z0-9]{18}[0-9]{2}`; rule "could validate against GLEIF; at minimum the format". | **Clarified.** A pattern failure cites the package. A check-digit failure cites ISO 17442 through the ITS `{LEI}` format, not the package. |
| 12 | Code lists are `xs:string` enumerations and dates are `xs:date`. | Confirms revision 6: padding is invalid around a code and collapsed around a number, boolean or date. |
| 13 | RREL65 is `[1..n]` of a choice: a date, or ND1, ND2, ND3 or ND5, or a dated ND4. | Confirms the RREL65 validator of revision 6. |
| 14 | The dated ND4 is a date plus the code. The package has no rule on that date. | P3e ("after the cut-off") rests on Article 9(3) alone and cites only that. |
| 15 | NUTS: pattern of two letters and three characters; a valid NUTS code, or a country code plus `ZZZ` where the country is not in the NUTS classification. Country: valid ISO code. | Confirmed. No change. |
| 16 | The RRE performing template has **no cross-field validation rule**. Every rule is on one field, on a date against the cut-off, or on earlier submissions. | The cross-column checks of section 7 rest on the field definitions, the Q&As and arithmetic, as already labelled. None can cite the package, apart from the cut-off comparisons. |
| 17 | The XSD has two optional elements that are not RTS fields: gross annual rental income for buy-to-let, current and at origination (`GrssAnlRntlIncm`, `[0..1]`, added in 1.3.0). The workbook lists them under the non-performing branch only, but the XSD type that carries them is shared by the performing branch. Each is an amount with currency, not below 0, or ND5; ND1 to ND4 are not offered. | Two optional-element entries in the catalogue (6.4, O1). When a column is supplied it is validated by these transcribed rules; its absence is never a finding. |
| 18 | The package defines the XML report only. | Header names, the `,`/`;` list separator, the leading minus and the ND spellings in a CSV remain this application's conventions, as stated in 5 and 6. |

### 4.3 Status of the catalogue, and the evidence kept

Source review and catalogue completion are separate statuses.

| Part | Status |
|---|---|
| Package (workbook 1.3.1, XSD 1.3.0), RRE performing template | reviewed in full; nothing to restart |
| Per-field source mapping | written: `docs/data-quality/rre-source-mapping.csv` beside this file as `rre-source-mapping.csv`, one row per field (107) and per optional element (2): workbook row numbers, XML path, multiplicity, XSD type and facets, ND options, validation rule, change-log amendment, CSV representation, implementing checks, result kind, compliance and rejection effect, provisional final-report category, limitations. The ESMA package itself is not in the repository. The file is the reviewed table behind `src/esma_milan/quality/catalogue.py`: a test keeps field names, formats, ND flags and final-report categories equal, and keeps the `implementing_checks` column equal to the rules the catalogue applies to each field, in both directions. |
| Reference lists | bundled under `src/esma_milan/quality/refdata/`, one code per line, with `manifest.json` recording the source, its address, the version or publication date and the SHA-256 of each source file. ISO 3166-1 alpha-2: Debian `iso-codes` 4.20.1 (LGPL-2.1-or-later, notice beside the file; codes only). ISO 4217: SIX list one (published 2026-09-17) and list three (published 2026-01-01). NUTS level 3 for 2010, 2013, 2016, 2021 and 2024: Eurostat's GISCO distribution service. The NUTS lists cover 35, 35, 37, 37 and 38 countries (the United Kingdom up to 2021 and not in 2024) and hold no `ZZZ` code. |
| ESMA Q&As | sweep done on 16 November (4.4) |
| Final-report (L1) field categories | assignment made for all 107 fields. No Q&A lists the "current" numeric fields or addresses RREL8 on a final report, so category 2 and the RREL8 treatment remain my reading of Q&A 1296 and the field definitions. That uncertainty is kept in the rules: L1 only ever raises questions, and ND5 in RREL8 on a repurchase is a question. |
| RREL71 on a never-defaulted loan (0 or ND5) | settled: ND5 compliant (Q&A 1324), 0 an accepted alternative (Annex text) |
| RREL77 | transcribed (C44); offset law is outside the files |
| Policy decisions | section 15 |

### 4.4 The Q&A sweep (16 November 2026)

Scope: the Q&As of ESMA's securitisation Q&A document (ESMA33-128-563,
last consolidated 13 July 2023) that concern all underlying exposure
templates or Annex 2: sections 5.1.11 to 5.1.27, 5.2, 5.3 and 5.4. I
read about 75 Q&A pages on the ESMA site in addition to the eight read
before. Not read: Q&A page 1321 (the page returned an error twice); the
answer to 5.2.6 (arrears breakdown; the page shows the question only);
5.4.8 (allocated losses and cumulative recoveries for Annex 2), which I
could not locate as a page and whose subject is covered by Q&As 1344 and
1348; anything published after the consolidated document. These are
stated limitations, not passes.

Q&As that change or source a rule:

| Q&A (document number) | Instruction | Effect |
|---|---|---|
| 1324 (5.2.5) | default amount and date: ND5 "in all other cases"; unchanged on re-performing | RREL71 never-defaulted branch settled |
| 1334 (5.2.14) | FLCF: RREL47 = ND5, switch date in RREL51. FXPR: months in RREL47, reset date in RREL51. FXRL: RREL47 = ND5 | C12(c) and C38 now sourced and violations; C13 basis |
| 1295 (5.1.14) | FLCF is reported until the loan becomes floating, then FLIF; origination data need not be re-collected, the rest reflects the cut-off | C12(c), C13 |
| 1344 (5.3.6), 1348 (5.3.12) | cumulative recoveries: ND5 with no recovery process, 0 with none recovered; allocated losses: ND5 before any work-out | RREL73, RREL74 rows; C43(c) stays a question |
| 1346, 1358 (5.3.9, 5.3.25) | date last in arrears: ND5 if never in arrears; this exposure only | C7 unchanged, now sourced |
| 1354 (5.3.20) | prepayments allowed: 0 where none are allowed without penalty | RREL59: zero is never a placeholder finding |
| 1307 (5.3.15) | grace period end: ND5 when none in effect at cut-off | C17(a) question, sourced |
| 1308 (5.3.8) | payment due: next scheduled payment; 0 during a payment holiday | C16 stays plausibility |
| 1353 (5.3.19), 1363 (5.3.29) | ND5 accepted for energy certificate fields when unavailable, and for the credit-impaired field in non-STS deals | ND5 never questioned on RREC10, RREC11, RREL14 |
| 1377 (5.4.7), 1357 (5.3.23) | every Annex 2 exposure has a collateral item; a sold property stays reported with RREC20, RREC21 and a valuation as at the sale date | F2: a loan with no collateral record is a compliance violation; E12 context |
| 1376 (5.4.6) | RREL16 excludes expected future rent; rental income goes in the optional rental-income fields; a deceased obligor's income is generally 0 | confirms O1 and "zero income can be real" |
| 1369 (5.3.35) | repurchase date is the date investors cease to have recourse | supports RREL8 as the event date on a repurchase; no ND5 instruction found |
| 1290, 1288, 1299, 1293 (5.1.11, 5.1.18, 5.1.12) | one ND option per field; ND5 only where not applicable; OTHR never replaces an ND option; no proxy data | P3 rules confirmed |
| 1297 (5.1.16) | no rounding, except amounts with more than 5 decimals, rounded to 5 | P5 facet confirmed |
| 1320, 1319, 1310 (5.2.3, 5.2.2, 5.1.26) | RREL3 is never blank and repeats RREL2 when unchanged; RREL1 survives a change of reporting entity; split loans keep the identifier with a suffix | identifier rules confirmed; a suffix such as `_split_1` is not a finding |
| 1312 (5.1.27) | a legal name over 100 characters is cut to 100 | P7(a) on RREL79, RREL82 confirmed |

Read and consistent with the plan, with no rule change: 1294 (further
loans share obligor and collateral identifiers), 1300 (original lender),
1302 (anonymity), 1303 (identifier method), 1306 (special scheme), 1309
and 1356 (account status precedence, arrears balance), 1314, 1316 (pari
passu amount, 0 when none), 1326, 1328, 1332, 1362 (index and floor
fields), 1327 (revision margins and dates in chronological order), 1336,
1337, 1365 (history has no time limit), 1338, 1342, 1350, 1364 (bullet
and balloon conventions), 1347 (credit limit only for flexible loans),
1359, 1360, 1361, 1366, 1368, 1371 (lock-out and fee end dates: ND5 when
none, latest date when several), 1372, 1373, 1374.

---

## 5. The audit read: lineage, value states, inventories

### 5.1 Reading without losing the original

The audit does not use the pipeline's cleaned frames, and a plain Polars
read is not an untouched record either (duplicate headers are renamed;
an unquoted empty cell becomes null while a quoted one stays an empty
string). So:

1. The header record is read on its own with the standard `csv` module:
   original header text and position of every column, before any
   normalisation. Duplicate header texts are seen here.
2. The body is read with every column as text and no null tokens. Null
   and empty string are both mapped to the state "empty"; the report says
   that quoted and unquoted empties are not distinguished.
3. Each record gets an immutable audit key: file and **logical CSV record
   number** (1 is the first data record). Quoted fields can contain line
   breaks, so this is not a physical line number; the report says so, and
   also gives the physical line where the record starts when it differs.
4. Encoding: the bytes are decoded as the pipeline does (`_utf8_source`);
   cells containing repaired bytes are flagged, and the original bytes of
   those cells are kept in hexadecimal for the detail file.

The audit frames never reach the pipeline, and the pipeline's inputs are
never replaced by audit-parsed or repaired values.

**As implemented** (`quality/audit_read.py`). The header is the first
non-empty record of the decoded text as the `csv` module reads it, with
a byte-order mark removed. The body is read by Polars from the same
bytes with every column as text and no null tokens: the parser the
loader uses, so the records are split exactly as the pipeline splits
them. Record n of the audit (1-based) is index n − 1 of
`PipelineResult.loan_record_index` and `collateral_record_index`; a
blank line after the header is a record, blank lines before the header
are not, and a quoted field that spans lines is one record. Columns are
addressed by position, never by name. The physical line a record starts
on is computed: the header's last line, one line per earlier record, and
the line breaks inside earlier cells. The audit decodes with its own
copy of the loader's steps (the loader's function logs an event) and
keeps, for every repaired cell, the record, the column position and the
original bytes in hexadecimal.

The file part of the audit runs before the pipeline, one file at a time,
and frees the file's text before reading the next, so the audit and the
pipeline never hold their frames at once.

### 5.2 What is kept for every record

Audit key; raw original and new loan, borrower and collateral
identifiers; their normalised forms (trimmed) with before/after evidence;
the identifiers the pipeline resolved (when Stage 3 ran); eligibility and
population (section 8); links to related records (a collateral record's
loan record or records; a loan's collateral records). Per cell: original
header and position, raw value, value state, parsed audit value, and
whether the parse succeeded.

Trimming can merge two different raw strings into one identifier. The
audit detects that (A13b) and, where identity is ambiguous (duplicates,
collisions, a collateral record matching two loan records), the report
states the ambiguity and gives ranges or "not determinable" instead of a
precise distinct-loan count or balance share.

**As implemented.** Per record the audit keeps the raw and trimmed
identifiers, the status, the balance text, the cut-off, the population,
the identity-confidence label and the claimed lifecycle (loans), and the
collateral type and class (collateral); and, for the waterfall and A17,
the balance as the loader reads it, the currency stated for each amount
the pipeline combines, and for a collateral record the reasons of Stage
2's collateral predicate that the record alone establishes. The
identifiers the pipeline resolved are not copied onto the records yet.

**The links, as implemented.** The audit builds the links between
collateral records and loan records from its own text, without calling a
stage. A collateral record refers to a loan through RREC2. Annex 2
prescribes RREC2 = RREL3 (convention `new`); exports also carry the
original identifier there (convention `original`, RREC2 = RREL2), and
Stage 3 chooses between the two. Each convention is matched twice and
the results are kept apart: `raw` (the two cells are equal as they
stand) and `trimmed` (equal only after trimming whitespace, which the
pipeline does not match, #46). An empty cell and an ND token are not
identifiers and link nothing. One row per (collateral record, loan
record, convention) is kept. From the links: the claimed lifecycle of
the loan a collateral record belongs to (P3f, 6.1), whether the record
belongs to a retained loan (the retained-pool value states of 6.4), the
causes of the waterfall's line 6 (8.2), and a record-by-record
comparison with the loans Stage 3 dropped whenever Stage 3 ran. The
summary gives the counts per convention and match and the records that
link to nothing.

### 5.3 Two inventories

**Expected-field inventory**: the 84 RREL and 23 RREC fields, one row
each, including absent fields (section 6.4).

**Actual-column inventory**: one row per physical column of each file, in
file order: position, original header text, how it was resolved (ESMA
code, taxonomy field name, cleaned name, currency companion of a field,
known export extension such as `Sec_Id` and `Pool_Cutoff_Date`, or
**unmapped**), the field it resolved to, and a generic profile (records,
empty, distinct values, longest value, sample values). Aliases are
resolved within the file's own schema (a loans header never resolves to
an RREC field). An unmapped column gets the profile and the statement "no
taxonomy mapping established; not assessed for compliance". It is never
shown as compliant. Currency companions get their own format and
membership rules (6.2).

Duplicate policy (replacing the conflict between A1 and A15). Three
cases, tested in this order:

| Situation | Outcome |
|---|---|
| The same header text twice | A1, Stop under a new **acceptance policy** (A): today the loader's parser accepts this and renames the later columns (`<text>_duplicated_<n>`), so the pipeline silently uses the first. Every column is profiled, and the inventory shows the name the pipeline gives each. |
| Two different headers that the pipeline gives one name (`RREL3` and `New Underlying Exposure Identifier`) | A1, Stop (L): the pipeline cannot process the file; it raises `duplicate_columns` today. |
| A field supplied in two columns, by either case above | A15 as well: equal values on every record, Warn (one field reported twice); values differing on any record, Stop under the acceptance policy (A), and every differing record is listed, two rows a record, one per column, each naming the other column's value. |

An earlier version of this table had a fourth row: two different headers
resolving to one field with identical values, as a Warn on its own. It
cannot occur. Every header the audit resolves to a field, by code, by
taxonomy name or by cleaned name, gets the same name from the pipeline,
so two different headers of one field always collide and are the second
case.

**How a header is resolved, as implemented.** Within the file's own
schema, in this order: (1) the header is exactly an ESMA code; (2) it
is exactly a taxonomy field name; (3) its cleaned name
(`clean_names`) equals a field's cleaned name; (4) it is an optional
ESMA element by this application's header names (6.4); (5) its cleaned
name is `<code>_currency` for a `{MONETARY}` field of the schema or for
an optional element; (6) its cleaned name is `sec_id` or
`pool_cutoff_date`; otherwise it is unmapped. A code is matched exactly,
as the pipeline matches it: `rrel30` and a padded `RREL30 ` are not
codes. The inventory also shows the column name the pipeline gives the
header, because the two differ by design in three places: a code or
name of the other file's schema (the pipeline renames it by the whole
taxonomy; the audit says unmapped); a companion the pipeline's fixed
lists do not know (`RREL30_CURRENCY`, which the pipeline keeps); and a
repeated header text (Polars renames the second column; the audit
resolves both to the field and assesses each column separately). The
duplicate policy above is a finding of family A and is not reported yet;
the inventory shows how each header resolved.

---

## 6. Part 1: per-column rules

Every field gets the same rule set, driven by its catalogue entry, on
every record of its file. Basis letters are those of 3.2.

### 6.1 Value states and the ND rules

Each cell is classified once, before any format rule, into one state, and
the states stay visible through the whole audit:

| State | Recognised as |
|---|---|
| Field absent | the field has no column in the file |
| Empty | empty or whitespace-only cell |
| Invalid missing marker | decided **after** the field's own domain, never before: a value that is valid for the field is a present value. `NA` in a `{COUNTRYCODE_2}` field is Namibia. In free-text fields nothing is classified as a missing marker (`NULL`, `-`, `NONE` can be content; P8 may raise them as possible placeholders). In typed fields (number, date, list, boolean, currency, NUTS, LEI), `NA`, `N/A`, bare `ND`, bare `ND4`, `ND6`, `nd5`, `ND 5`, `NULL`, `None`, `-`, `#N/A` are invalid missing markers. |
| ND1, ND2, ND3 | exactly those tokens |
| ND4 with date | `ND4-YYYY-MM-DD` (Article 9(3)); the date is parsed and kept |
| ND5 | exactly `ND5` |
| Present value | anything else; goes on to the format rules |

Classification order, the same for every field including free text:

1. Exact prescribed tokens `ND1`, `ND2`, `ND3`, `ND5` are ND states in
   every field. They are never accepted as text content, so the ND
   allowance rules (P3a, P3b) always apply.
2. ND4. Well-formed `ND4-YYYY-MM-DD` with a real date is the ND4 state
   in every field. Otherwise the field's own domain is tested before
   any malformed-ND4 conclusion: a value valid for the field is a
   present value. The prefix shortcut is used only for the types in
   which no valid value can start with `ND4` (number, date, year,
   boolean, list, currency, country and NUTS codes): there, bare `ND4`
   and any text starting `ND4` is a malformed ND4 (P3e). `{LEI}` is not
   one of those types: `ND4X0000000000000085` has the ISO 17442 shape
   and valid check digits, so it is a present LEI, and only an `ND4...`
   value that fails the LEI rule is examined as an ND4 attempt, by the
   same narrow test as text fields. In
   `{ALPHANUM-n}` fields a prefix proves nothing (`ND4000123`, `ND4XYZ`,
   `ND4-ALPHA` are ordinary identifiers): only the exact token `ND4` and
   text matching `ND4-` followed by a date-shaped string of digits and
   hyphens (`ND4-2027-13-01`) are treated as ND4 attempts; everything
   else starting `ND4` is content.
3. A value valid for the field's own domain is a present value (`NA` in
   a country field).
4. In typed fields, the remaining look-alike markers are invalid missing
   markers (P3d).
5. In free-text fields everything else is content, including `NULL` and
   `-`; only P8 may raise a question about it.

A valid ND value is never also reported as a failed number or date.

**Settled readings of the classification** (as implemented in
`quality/states.py`):

- A cell is *empty* when it holds no character other than Unicode
  whitespace. (Numbers, booleans and dates are trimmed of XML whitespace
  only, 6.2.)
- "Exact" means exact. A padded ND token (` ND5`), and a dated ND4 that
  is well-formed only after trimming XML whitespace, is not an ND state:
  in a typed field it is an invalid missing marker (P3d); in a free-text
  field it is content.
- The invalid missing markers are a closed list
  (`DQ_INVALID_MISSING_MARKERS`: `NA`, `N/A`, `#N/A`, `NULL`, `NONE`,
  `-`, `ND`) plus `ND` followed by an optional space and one digit when
  that is not an exact valid token (`nd5`, `ND 5`, `ND6`). They are
  compared after trimming XML whitespace and without regard to case, in
  typed fields only, and always after the field's own domain. `NaN` is a
  format failure (P5), not a marker.
- Bare `ND4` is a malformed ND4 (P3e) and not also an invalid marker:
  step 2 comes first.
- A malformed ND4 in a field that allows no ND4 is P3e only. A
  well-formed dated ND4 there is P3a, and also P3e when its date is not
  after the cut-off (two defects).
- For a country or currency code, "valid for the field" includes
  membership of the reference list, so `ND` in a country field is a
  marker and `NA` is Namibia. A code of the right shape that is not in
  the list (`XX`) is a present value: its format passes P5 and P11
  reports it.
- An RREL65 cell that holds a `,` or `;` is a present value and goes to
  the RREL65 validator before any ND4 prefix test.
- The inventory's nine state columns are: empty, invalid marker, ND1,
  ND2, ND3, dated ND4, malformed ND4, ND5, present.

**The cut-off date** used by P3e, P6(b), P12 and the lifecycle label.
For a loan record: RREL6 when it is a valid date, else the export's
`Pool_Cutoff_Date` column; when both are valid and differ, the cut-off
is not established for that record and the rules that need it are not
evaluable there. For a collateral record: the file's own
`Pool_Cutoff_Date`; where the file has no such column, the loans file's
cut-off when its records give exactly one, and the result row then says
"pool cut-off taken from the loans file"; otherwise not established. A9
(one cut-off across the files) is a finding of family A.

| Id | Rule | Basis | Severity |
|---|---|---|---|
| P1 | **Field absent from the file.** Every one of the 107 fields is required to be reported (Q&A 1283: an ND allowance does not make a field optional). One finding per absent field, whether or not MILAN uses it. Separately, if the pipeline requires the column, A2 is the Stop. | E (and L via A2) | Warn |
| P2 | **Value-state profile.** Counts of each state over all records and over the retained active pool. | P (context) | Info |
| P3a | ND1 to ND4 used where Annex 2 does not allow them. | E | Warn |
| P3b | ND5 used where Annex 2 does not allow it. | E | Warn |
| P3c | Empty cell, in any field. A missing value must be a present value or a permitted ND option. | E | Warn |
| P3d | Invalid missing marker, as classified above (field-aware). | E | Warn |
| P3e | ND4 date: not a real calendar date, or not after the data cut-off date. | E | Warn |
| P3f | **Conditional ND5.** Where the field definition ties applicability to other supplied values, ND5 is checked both ways: ND5 present where the field is established as applicable, and a value present where the definition prescribes ND5. The conditional rules are listed per field in 6.3 (for example RREC7, RREC8, RREC9, RREC12, RREC16 on non-property collateral). Where nothing in the files establishes applicability the ND5 is reported as "allowed; justification not verifiable from the files". | E | Warn |
| P3g | **What the pipeline does with the state.** The loader treats empty, `NA`, `ND`, `ND1` to `ND5` as missing in every column (`CSV_NA_TOKENS`), which also turns the valid country code `NA` into missing: the source value is valid and the loader loses it, reported as L, not as a reporting fault. It does not recognise `ND4-YYYY-MM-DD`: in a numeric field the loader raises, which is A5's row and not a P3g row (one defect, one row); in a date field the token survives the read and the date parser then sets it to missing with a warning; in a text field it stays as text. | L | Warn |

The pipeline's handling of the dated ND4 form is a defect of its own
(section 12): a compliant file can be unprocessable today.

**Settled readings of the ND rules** (as implemented):

- **P3f.** The conditional rules are RREC7, RREC8, RREC9, RREC12 and
  RREC16 against RREC5. Established non-property with a value: violation
  (RREC8 on a GUAR record excepted, Q&A 1323(c)). OTHR, MIXD, or an
  RREC5 that is not a valid code: limitation. A property row with ND5 is
  decided through the link to the record's loan (5.2), by the lifecycle
  that loan claims (3.3):

  | The linked loan's claimed lifecycle | ND5 on a property row |
  |---|---|
  | final-inactive (presumed) | no row: ND5 is the prescribed value |
  | reporting-active (presumed) | question: the field is applicable to a reporting-active exposure |
  | default, recoveries not determinable | limitation, with that reason |
  | not established | limitation |
  | the record links to loans whose lifecycles differ | limitation |
  | no loan record linked | limitation |

  The lifecycle is taken under the prescribed convention on the raw text
  first, else under the other. RREC12 = 0 on a negative loan balance
  needs the loan's balance beside the collateral record and is E9's. ND5 in a field with no conditional rule produces no row:
  it is the scope statement "allowed; justification not verifiable from
  the files", printed once, with the ND5 count per field in the
  inventory. The "mixture not explained" question of RREL65 (6.3) is
  emitted under P3f.
- **P3g** has one row per cell where the loader does something other
  than what the file states: a value valid for the field that the loader
  reads as missing (`NA` in a country field; `NA` or `ND` as free
  text); a dated ND4 in a date or text field (in a numeric field the
  loader stops, and the row is A5's); a date in a form the pipeline
  parses although it is not `YYYY-MM-DD`; and, for an RREL65 cell with
  several elements, the date the pipeline's split keeps. The last two
  are the "(L)" notes of 6.2 and 6.3.
- **Severity Stop without a decision.** Until the acceptance rules are
  implemented, a Stop row says what it observes and by which basis: "the
  pipeline cannot process this" (L) or "would stop under the acceptance
  policy" (A). A rule whose severity is Stop has no effect on the run,
  and the report never says a pool was blocked or rejected.

### 6.2 Format rules by type

Validation uses the raw string with exact decimal arithmetic
(`decimal.Decimal`), never a binary float.

| FORMAT | Compliant (E, source) | Notes |
|---|---|---|
| `{MONETARY}` | One validation path for numbers, in this order: (1) keep the original cell as evidence; (2) apply only the normalisation the type permits: XML whitespace collapse, i.e. removing leading and trailing space, tab, carriage return and line feed, and no other Unicode whitespace (not a general `strip()`); (3) lexical check of the result against the `xs:decimal` form: optional sign, then ASCII digits with an optional `.`, with at least one digit somewhere, so `12`, `.5`, `1.`, `+01.50` pass and `1_000`, `1e3`, full-width digits, `NaN`, thousands separators and currency symbols fail; (4) value facets with exact decimal arithmetic: at most 18 digits in total and 5 fractional, counted on the value (`1.234560000` passes, `1.234567` fails). | The package holds an amount as an unsigned decimal plus a sign indicator; a leading `-` in the CSV is this application's representation of that indicator and passes the lexical check. Which fields may be negative is P6. Surrounding XML whitespace is therefore **not** a P5 violation; P4(a) alone reports it, as a CSV-convention note. The amount's currency is in the companion column. Sign is not a format matter (P6). The same path serves `{PERCENTAGE}`, `{INTEGER-n}` and, with the boolean literals, `{Y/N}`. |
| `{PERCENTAGE}` | a decimal with at most 11 digits in total and at most 10 fractional digits, in percentage units (`7.0` is seven per cent). | No general 0 to 100 range. A suspected decimal-fraction export is a plausibility finding (P12), and the checker never reinterprets the values. |
| `{INTEGER-9999}` | an integral value from 0 to 9999 (XSD `Max4PositiveNumber` for the five RRE fields: RREL25, RREL47, RREL58, RREL68, RREC8). `12.0` is integral and passes; `12.5` fails. | |
| `{DATEFORMAT}` | `YYYY-MM-DD`, a real calendar date, after the same XML whitespace collapse. **RREL65 is dispatched to its own validator before this rule** (6.3): the cell is a list of dates and the single-date rule is applied to each element, never to the whole cell. | `YYYY/MM/DD`, a trailing time and an Excel serial in 1900 to 2100 are **not** the prescribed format and get the format finding (E); separately the report records that the pipeline parsed them (L). Anything else is also set to missing by Stage 1. Plausibility windows are separate (P12). |
| `{YEAR}` | `YYYY`. | For RREL12, also a NUTS classification year (P11). |
| `{Y/N}` | Three things are reported per value: the source spelling, the interpreted boolean, and conformity. `Y`, `N`: the ITS template representation, compliant. `true`, `false`, `1`, `0`: valid `xs:boolean` literals, the XML representation; an accepted alternative, not a violation. `TRUE`, `FALSE`, `Yes`, `No`, `y`, `n` and anything else: neither form, Warn, with the interpretation shown when unambiguous. | Mapper compatibility is separate (L) and follows the whole path: raw spelling, the dtype Polars infers for the column, the coercion to text, the mapper. A column whose non-null values are all `true`/`false` infers as Boolean (ND tokens become null and do not prevent this), is coerced to `TRUE`/`FALSE`, and Borrower Residency then reads `Y`/`N`; a column holding any other non-null token that prevents Boolean inference stays String, and each cell is then compared as text: upper-case `TRUE`/`FALSE` cells still read `Y`/`N`, any other spelling reads "ND". Recourse compares with `Y`/`N`, so a Boolean column reads "ND" (#47). The report states the path it observed for the column. |
| `{LIST}` | exactly one code from the field's reviewed code list, with no surrounding whitespace (the code types do not collapse it). | Values that would match after trimming or upper-casing are violations and are counted separately, so the cause is visible. What MILAN makes of each code comes from the actual code map (9.3). |
| `{NUTS}` | five characters; a NUTS level 3 code of the classification year in RREL12, or a valid ISO 3166-1 country code followed by `ZZZ` where no NUTS 3 exists. | Membership needs the Eurostat list for the stated year (P11). If RREL12 is missing or the year's list is not bundled, membership is "not evaluated" and only shape is checked, labelled as a limited check. |
| `{COUNTRYCODE_2}` | two capitals, a current ISO 3166-1 alpha-2 code. | P11 |
| `{CURRENCYCODE_3}` | three capitals, an ISO 4217 code. | P11 |
| `{LEI}` | 20 characters, ISO 17442 shape, valid ISO 7064 MOD 97-10 check digits. The package pattern is `[A-Z0-9]{18}[0-9]{2}`; the check digits come from ISO 17442 and each failure cites its own source. | Shape and checksum do not prove the LEI belongs to the named entity: "entity match not assessed". |
| `{ALPHANUM-n}` | at most n characters, ASCII, capital letters without accents (ITS Annex I). The XSD restricts only the length (1 to n), so a lower-case value is an ITS format finding that the schema would accept. Free text: spaces and punctuation are legitimate. | Owned by P7 alone (length and character set); P5 does not test `{ALPHANUM-n}` fields, so a lower-case value cannot reach the MILAN decision through a second rule. |

Currency evidence. Every `{MONETARY}` field's definition says to
include the currency, and the package requires a valid currency on every
amount (4.2). The regulatory requirement is that the currency is
**stated**; which CSV column states it is this application's layout.
For each monetary value present the report establishes one of five
cases, and a successful number-format check never covers any of them:

| Case | Evidence | Result |
|---|---|---|
| 1 | the amount's own companion column holds a valid ISO 4217 code | stated; nothing reported |
| 2 | no usable companion, but a dedicated field with a documented mapping supplies it: RREL28 for RREL29 and RREL30; RREL18 for RREL16; RREC22 for RREC13 | stated. If the companion column is physically absent, that is a layout note under this application's CSV contract (basis A, Info), not a compliance violation |
| 3 | companion and dedicated field both supplied and different | conflict, owned by C30 (X) |
| 4 | neither supplies a valid code (absent, empty, ND, or not an ISO code) | the currency is not stated: compliance violation (E); the unit is unknown for every calculation (question 8) |
| 5 | a column this application's CSV contract expects is absent while the currency is stated another way | layout note (A), as in case 2 |

The mappings stay narrow. RREC22 describes RREC13 and says nothing
about RREC17. RREL18 describes primary income and is not extended to
RREL20. RREL28 is not extended to other loan amounts until a basis is
documented; RREL77 is compared with the exposure's currency because its
definition says so (C30).

**Settled readings of the format rules and currency evidence** (as
implemented):

- P4(a) covers `{YEAR}` as well: `xs:gYear` collapses whitespace.
- A control character inside a typed value fails the P5 lexical check;
  P5 emits the one row, with the reason "control_character", and P4(c)
  shares its defect key and emits nothing, so the defect counts on a
  critical field under P5.
- P4(d) applies to the `{LIST}` fields whose values the pipeline reads
  (Appendix B), for a value that is a list code only after trimming or
  upper-casing. Identifier padding is A13.
- A date is exactly `YYYY-MM-DD`; the time-zone suffix `xs:date` allows
  is not accepted.
- `{INTEGER-9999}`: `+12`, `0012`, `12.0` and `-0` pass; a negative
  value fails P5 (the type's own facet), not P6.
- Digits are counted on the value: leading zeros and trailing fractional
  zeros are not digits.
- P13 is evaluated for every amount that passes P5. A companion cell
  gets its own format (P5) and membership (P11) results only where its
  amount is present; an empty or ND companion is not a finding of its
  own. Until C30 exists, a companion and a dedicated field that differ
  are case 1 (the companion states the currency).

Remaining per-column rules:

| Id | Rule | Basis | Severity |
|---|---|---|---|
| P4 | Whitespace and characters. Three facts are kept apart for every padded value: whether the source type permits the padding; this application's CSV convention; the pipeline effect. (a) Leading or trailing XML whitespace around a value whose XSD type collapses it (decimal, boolean and date; confirmed in the XSD, 4.2): the padded value is valid for the type, so it is an **accepted alternative**, and treating it as nonconforming is only this application's CSV convention (basis A, labelled as such). (a2) Padding around a code-type value (`{LIST}`, currency, country, NUTS, LEI): the code types do not permit it, so ` PERF` or ` GB ` is a **P5 violation** and P4 emits nothing for it; one result, one defect key. (b) Whitespace inside or around free text: not a finding. (c) Control characters or line breaks inside a typed value: violation of the field's format (E). (d) The processing effect, recorded separately as L where an exact comparison in the pipeline is defeated (a status of ` PERF` is not active; section 8). | A / E / L as stated | Warn or Info |
| P5 | Format conformity per the table above, for every type except `{ALPHANUM-n}` (P7). | E | Warn |
| P6 | **Value domain stated in the field definition or a published validation rule**, only where one exists. From the package (4.2): (a) not below 0 for RREL16, RREL20, RREL29, RREL31, RREL32, RREL33, RREL41, RREL61, RREL64, RREL67, RREL71, RREL77, RREC13, RREC17, RREC21; RREL30, RREL39, RREL73 and RREL74 may be negative; (b) not later than the cut-off for RREL7, RREL8, RREL9, RREL23, RREL63, RREL65 (each date element), RREL66, RREL72, RREC15, RREC19, RREC20; no such rule for RREL24, RREL36, RREL51, RREL53, RREL55, RREL60, RREL62; (c) RREL59 between 0 and 100; (d) RREL12 year bounds; (e) RREL6 in the past, owned and reported once by A10(a), which does not count towards rejection (8.3). Counting on critical fields is set out in 8.3. From the field definitions: RREC12 is 0 when the balance is negative; RREL46 may be negative. These apply whether or not a row of 6.3 repeats them, and where 6.3 tags one of the eleven cut-off comparisons [X] it now carries the regulatory source. No blanket sign or range rule by type. Default amount and date follow the lifecycle rule in C10 and are not a P6 example. | E or X | Warn |
| P7 | Length and character set for `{ALPHANUM-n}`, as separately keyed facets: (a) longer than n characters, counted on the cell as read; (b) not ASCII capitals: any character outside the range space to tilde, or a lower-case letter; (c) for RREL1 and RREC1 only, the Article 11(1) construction: the reporting entity's LEI (shape and ISO 17442 check digits), `N`, a four-digit year and a two-digit number, 27 characters in all. | E | Warn |
| P8 | **Possible placeholder.** A value that often stands for "not applicable" but can also be a real measurement: `0` in an income, valuation, price or limit field; `999.99999`, `9999`; a single token such as `N` filling a free-text field. Worded "possible placeholder; confirm whether this is a measurement". Zero income, a zero prepayment allowance and zero sale proceeds can be real. Nothing is converted to ND5. | P | Warn where the field feeds MILAN, else Info |
| P9 | Cardinality: constant columns, code distributions. | P | Info |
| P10 | MILAN lineage: what the output shows for this field's values (section 9.3). | L | Info |
| P11 | Reference-list membership: ISO 3166-1, ISO 4217, NUTS 3 for the RREL12 year. When the reference version needed is not bundled the result is "limited assessment: shape only", never "invalid code". The ESMA currency type permits active and historic ISO 4217 codes, so a historic code is valid. Any further timing remark (a withdrawn currency on an amount whose own date, origination, valuation, sale or cut-off as the field dictates, is after withdrawal) is plausibility only, and only once that date is defined for the field. RREL12: year greater than 1980 and less than 2100 (validation rule); when every RREL11 and RREC6 value is a country-plus-`ZZZ` code, any year is accepted (Q&A 1313). | E | Warn |
| P12 | Plausibility windows: dates before 1950 or more than 60 years after cut-off; a percentage column whose non-missing values are all below 1 (suspected fraction export); field-specific windows in 6.3. | P | Warn or Info |
| P13 | **Currency evidence**: the currency of every amount is stated (the five cases above). A violation is case 4; the layout note of cases 2 and 5 is one context row per absent companion column (basis A, Info); case 3 is C30's. Evaluated for every amount that passes P5; an amount that does not parse is not evaluable. | E | Warn |

**Settled readings of the remaining rules** (as implemented):

- **P2 and P9 are profiles.** P2 has no result rows: it is the
  value-state counts of the inventory. P9 emits one context row per
  column with one value on every record that reports a value.
- **P6.** (e) is owned by A10(a) and is reported with family A. "RREC12
  is 0 when the balance is negative" needs the loan and arrives with E9.
- **P8**, the list (`DQ_POSSIBLE_PLACEHOLDERS`). Per cell: `0` in
  RREL16, RREL34, RREC13 and RREC17; `9999` in the five
  `{INTEGER-9999}` fields; `999.99999` in RREC12 and RREC16. Per
  column, one row: every reported value is `0` in RREL33; every reported
  value is `0` in RREC21 while RREC20 holds no date; one token without
  a space on every record of RREL22. Never RREL20 or RREL59. Severity
  Warn where the field feeds MILAN (Appendix B), else Info.
- **P11.** Any ISO 3166-1 code or NUTS country prefix followed by `ZZZ`
  is accepted (`GBZZZ`, `ATZZZ`, `ELZZZ`); anything else before `ZZZ` is a
  violation. `EL` and `UK` are Eurostat's prefixes, not ISO codes, and
  are valid in NUTS codes of the years that hold them. RREL11 is tested against the record's own RREL12;
  RREC6 against the loans file's RREL12 when that is one valid year. A
  year that is not bundled, a missing year, or a country the bundled
  year does not cover gives "limited assessment: shape only", never
  "invalid". More than one RREL12 year in the file is one violation
  about the submission. Q&A 1313 is tested over both files. For an LEI,
  P11 would need the GLEIF register and is outside the files.
- **P12.** Date windows apply to valid dates of every `{DATEFORMAT}`
  field. "Every value below 1" applies to every `{PERCENTAGE}` field
  except the four margins (RREL46, RREL50, RREL52, RREL54) and RREL59,
  needs at least one non-zero value, and uses absolute values. RREC8 = 0
  ("not a position") is a P12 question.
- **RREL65.** A cell without `,` or `;` takes the generic path.
  Elements are trimmed of XML whitespace. An empty element (a trailing
  separator) is an invalid fragment.
- **Y/N mapper compatibility** (6.2) is an output-lineage statement and
  arrives with P10.

### 6.3 Every field

"Field-specific rules" lists what applies beyond P1 to P12, each tagged
with its basis; ids in parentheses are section 7 checks. "Pipeline" is
what the code does with the column today (corrected against
`config.py`, `filters.py`, `milan_map.py`). "Crit." marks fields on the
critical list (section 8.3).

#### Loans file

| Code | Field | FORMAT | ND1-4 | ND5 | Field-specific rules | Pipeline | Crit. |
|---|---|---|---|---|---|---|---|
| RREL1 | Unique Identifier | ALPHANUM-28 | NO | NO | [E] Article 11(1) construction: reporting entity's LEI, `N`, four-digit year, two-digit sequence; the LEI part passes the checksum. [X] One value in the file and equal to RREC1 (A11). The reporting entity need not be the originator, so no comparison with RREL83. | Dropped at Stage 1. | no |
| RREL2 | Original Underlying Exposure Identifier | ALPHANUM-1000 | NO | NO | [X] Unique per record (A7). [L] Whitespace, collisions with RREL3 of other records (A12, A13). Stability across periods not assessable. | Required column. Candidate loan id in Stage 3. | yes |
| RREL3 | New Underlying Exposure Identifier | ALPHANUM-1000 | NO | NO | [E] RREC2 must match this field (F3a). [X] Unique per record (A7). | Required column. Candidate loan id; MILAN Additional data 21. | yes |
| RREL4 | Original Obligor Identifier | ALPHANUM-1000 | NO | NO | [X] Mapping to RREL5 unambiguous (D2). A different RREL4 and RREL5 on one record is expected when an identifier changed. | Not in the Stage 1 minimum-column list, but identifier generation reads it (`original_id_col`), so its absence can fail Stage 3: needed for processing. MILAN Additional data 24. | yes |
| RREL5 | New Obligor Identifier | ALPHANUM-1000 | NO | NO | As RREL4. | Required column. Borrower id; MILAN Borrower Identifier, Additional data 23. | yes |
| RREL6 | Data Cut-Off Date | DATEFORMAT | NO | NO | [A] One value across the column, equal to the export's `Pool_Cutoff_Date` and to the collateral file (A9, Stop). [E] Not after the assessment date (workbook: "must be in the past"; boundary in A10(a)). [P] Not before every origination date (A10). | Dropped at Stage 1; the pipeline reads `pool_cutoff_date` (required column). | yes |
| RREL7 | Pool Addition Date | DATEFORMAT | NO | YES | [X] Not before RREL23, not after cut-off (C4). Used by C26. | Not mapped. | no |
| RREL8 | Date Of Repurchase | DATEFORMAT | NO | YES | On a repurchase status the date is the expected value, also on a final report (3.3, category 1); ND5 there is a question (C27). A date on any other status: question. [E] Not after cut-off (P6(b)). | MILAN Repurchase Date. | no |
| RREL9 | Redemption Date | DATEFORMAT | NO | YES | Q&A 1341: ND5 on an active exposure; the redemption date on RDMD; the completed-recovery date on a default status whose recovery process has ended; ND5 on a repurchase or other inactive case. [X] Missing or ND5 on RDMD (C27); not after cut-off. A valid date on RDMD or on a completed recovery is the prescribed value and is not an L1 question. | MILAN Redemption Date. | no |
| RREL10 | Resident | Y/N | YES | NO | [X] Against the country of the obligor's region and the country of the collateral, where assessable (C28). Boolean spelling per 6.2. | MILAN Borrower Residency: `Y`/`N` when the value reaches the mapper as `TRUE`/`FALSE`, which depends on the column's inferred type (6.2); "ND" otherwise. | no |
| RREL11 | Geographic Region - Obligor | NUTS | YES | NO | [E] NUTS 3 of the RREL12 year, or country code plus `ZZZ` (P11). | MILAN Borrower Postal/Zip Code, as text. | no |
| RREL12 | Geographic Region Classification | YEAR | YES | NO | [E] Greater than 1980 and less than 2100. [E] One classification year across the submission; a NUTS classification year, except when every region is a `ZZZ` code (Q&A 1313). | Not mapped. | no |
| RREL13 | Employment Status | LIST | YES | NO | [P] NOEM (legal entity) with an individual's income type, and the reverse (C20b). | MILAN Employment Type, Borrower Type. | no |
| RREL14 | Credit Impaired Obligor | Y/N | NO | YES | The field is a confirmation that the exposure was **not** impaired at selection. No comparison with current status. ND5 is accepted for non-STS securitisations (Q&A 1363) and is never questioned. [P] Count of records not confirmed (Info). | Not mapped. | no |
| RREL15 | Customer Type | LIST | YES | NO | Distribution only. | Not mapped. | no |
| RREL16 | Primary Income | MONETARY | YES | NO | [P] Below 1,000 or above 10,000,000 "unusual" (C20a); `0` is a possible placeholder and can be a real value (P8). Companion currency against RREL18 (C30). | MILAN Income Borrower 1, Total Income. | no |
| RREL17 | Primary Income Type | LIST | YES | NO | [P] CORP against RREL13 (C20b). | MILAN Borrower Type. | no |
| RREL18 | Primary Income Currency | CURRENCYCODE_3 | YES | NO | [X] Equal to the RREL16 companion currency (C30). A currency different from the loan's is information (FX exposure), not an error. | Not mapped. | no |
| RREL19 | Primary Income Verification | LIST | YES | NO | Distribution only. | MILAN Income Verification For Primary Income. | no |
| RREL20 | Secondary Income | MONETARY | YES | YES | [X] Pairing with RREL21 (C31). Zero is a measurement, not ND5. | MILAN Income Borrower 2, Total Income. | no |
| RREL21 | Secondary Income Verification | LIST | YES | YES | [X] A verification code while RREL20 is ND5 (no secondary income) (C31). A code with RREL20 = 0 is consistent (a verified zero). | MILAN Income Verification For Secondary Income. | no |
| RREL22 | Special Scheme | ALPHANUM-10000 | YES | YES | [P] A single token such as `N` on every record: possible placeholder for ND5 (P8). | MILAN Social Programme Type: recognised scheme text maps to "1" to "4" (right to buy, tenant purchase, help to buy or equity loan, shared ownership); missing reads "6"; any other present value reads "5", so a placeholder `N` reads "5" on every loan. | no |
| RREL23 | Origination Date | DATEFORMAT | YES | NO | [E] Not after cut-off (validation rule; C3). [X] Before RREL24 (C3). [P] One date shared by more than half the loans is unusual (C32). | Required column. MILAN Origination Date; seasoning (with the cut-off). | yes |
| RREL24 | Maturity Date | DATEFORMAT | NO | YES | [X] After RREL23 (C3). [P] On or before cut-off on an active loan; more than 50 years after cut-off (C3). | MILAN Maturity Date. | yes |
| RREL25 | Original Term | INTEGER-9999 | YES | YES | [P] Months, against RREL23 to RREL24 (C5; never counts). | Not mapped. | no |
| RREL26 | Origination Channel | LIST | YES | YES | Distribution only. | MILAN Origination Channel (all codes mapped, OTHR to "7"). | no |
| RREL27 | Purpose | LIST | YES | NO | Distribution only. | MILAN Loan Purpose (all codes mapped). | no |
| RREL28 | Currency Denomination | CURRENCYCODE_3 | NO | NO | [X] Equal to the RREL30 and RREL29 companion currencies on the record (C30). More than one currency across the pool is information for the verdict arithmetic (8.5), not a breach. | Not mapped; MILAN Loan Currency is the RREL30 companion. | no |
| RREL29 | Original Principal Balance | MONETARY | YES | YES | [P] Below RREL30 "unusual; capitalisation or further advances can explain it" (C1). [E] Negative: violation (workbook rule, P6). | MILAN Loan OB. | yes |
| RREL30 | Current Principal Balance | MONETARY | NO | YES | [E] May be negative (the workbook removed the non-negativity rule); no sign finding. [L] Zero, negative, missing or unparseable on an active status: excluded at Stage 2, each case counted separately (section 8). | Required column. Stage 2 filter (above 0); MILAN Loan CB; allocation. | yes |
| RREL31 | Prior Principal Balances | MONETARY | YES | YES | [E] 0 when there are none (definition); negative is a contradiction. [A] Equal to the loan's own RREL30 (C25, critical by decision; unusual, not impossible). | MILAN Total Prior Ranks CB, Ranking; External Prior Ranks CB reads "ND" when missing. | no (C25 is) |
| RREL32 | Pari Passu Underlying Exposures | MONETARY | YES | YES | [E] 0 when there are none; negative is a contradiction. | MILAN Pari Passu Ranking Loans (Not In Pool); reads "0" when missing (C18). | no |
| RREL33 | Total Credit Limit | MONETARY | NO | YES | [E] "Only to be populated for exposures with flexible or further drawing characteristics": a `0` on every record is a possible placeholder for ND5 (P8, P3f not verifiable). [P] Above 0 and below RREL30 (C34). | MILAN Flexible Loan Amount: limit minus balance, floored at "0", "0" when missing. | no |
| RREL34 | Purchase Price | PERCENTAGE | NO | YES | [P] Other than 100 (C23, Info); `0`. | Not mapped. | no |
| RREL35 | Amortisation Type | LIST | YES | NO | [X] BLLT against RREL41 (C35). | MILAN Principal Payment Type; OTHR reads "ND" (one of the few codes that do). | no |
| RREL36 | Principal Grace Period End Date | DATEFORMAT | NO | YES | [P] A date before cut-off: the definition asks for the date "if applicable as at the data cut-off date", so the report asks whether ND5 was intended (C17a, Info, not a compliance finding; Q&A 1307: ND5 when no grace period is in effect at the cut-off, and for an interest-only loan the end of the loan term). [X] After RREL24 (C17b). | Not mapped. | no |
| RREL37 | Scheduled Principal Payment Frequency | LIST | NO | YES | Distribution. Used by C26. | MILAN Principal Payment Frequency (all codes mapped). | no |
| RREL38 | Scheduled Interest Payment Frequency | LIST | NO | YES | Distribution. | MILAN Interest Payment Frequency. | no |
| RREL39 | Payment Due | MONETARY | NO | YES | [E] May be negative (no sign rule). [P] Zero on an amortising active loan; above RREL30 (C16); far from an annuity estimate (C37, Info). | MILAN Periodic Payment. | no |
| RREL40 | Debt To Income Ratio | PERCENTAGE | YES | YES | The field is a percentage of debt over income; its units are never reinterpreted. Negative: context or a question, never a violation on the sign alone (C24). [P] Pool-level: a median far below 100 suggests another scale or definition (C21). Recomputation only where the allocation basis is established (C47). | MILAN DTI Ratio, as given. | no |
| RREL41 | Balloon Amount | MONETARY | YES | YES | [X] Above RREL30 (C2); pairing with RREL35 (C35). | Not mapped. | no |
| RREL42 | Interest Rate Type | LIST | NO | YES | [X] The rate-type matrix (C12), written from the code meanings: FLIF, FINX, FLFL, CAPP, FLCA are floating; FXRL is fixed for life; FXPR and FLCF are fixed now with a future reset or switch. | MILAN Interest Rate Type (all codes mapped, OTHR to "8"). | yes |
| RREL43 | Current Interest Rate | PERCENTAGE | NO | YES | [P] Zero, negative or above 20 on an active loan: unusual, can reflect the product or credit risk (C11). | MILAN Interest Rate (divided by 100). | yes |
| RREL44 | Current Interest Rate Index | LIST | NO | YES | [X] Missing on a floating type; present on FXRL (C12). [L] LIBO or EURI without a tenor reads Base Index "ND" (C14). | MILAN Base Index. | no |
| RREL45 | Current Interest Rate Index Tenor | LIST | NO | YES | [X] Present without an index (C39). | MILAN Base Index. | no |
| RREL46 | Current Interest Rate Margin | PERCENTAGE | NO | YES | Negative is permitted by the definition (rate below index). [P] Above RREL43 is possible with a negative index (C15, Info). [X] Present on FXRL (C12). | MILAN Full Margin. | no |
| RREL47 | Interest Rate Reset Interval | INTEGER-9999 | NO | YES | Q&A 1334: ND5 on FXRL and on FLCF; the months between resets on FXPR (C38). | Not mapped. | no |
| RREL48 | Interest Rate Cap | PERCENTAGE | NO | YES | [X] Below RREL49; below RREL43 (C40). [X] Present with a type without a cap, missing on CAPP or FLCA (C40). | Not mapped. | no |
| RREL49 | Interest Rate Floor | PERCENTAGE | NO | YES | [X] Above RREL43; missing on FLFL or FLCA (C40). | Not mapped. | no |
| RREL50 | Revision Margin 1 | PERCENTAGE | YES | YES | Contractual revisions only, not periodic index resets; may be negative as RREL46. [X] Pairing with RREL51 (C15). | MILAN Margin After Reset Date. | no |
| RREL51 | Interest Revision Date 1 | DATEFORMAT | YES | YES | "Date interest rate next changes ... not the next index reset date". [X] After cut-off, not after RREL24 (C13). [P] Missing on FXPR or FLCF, whose type implies a future change (C12c). Never required for an ordinary floating loan. | MILAN Interest Reset Date. | no |
| RREL52 | Revision Margin 2 | PERCENTAGE | YES | YES | [X] Pairing with RREL53 (C41). | Not mapped. | no |
| RREL53 | Interest Revision Date 2 | DATEFORMAT | YES | YES | [X] After RREL51; not after RREL24 (C41). | Not mapped. | no |
| RREL54 | Revision Margin 3 | PERCENTAGE | YES | YES | [X] Pairing with RREL55 (C41). | Not mapped. | no |
| RREL55 | Interest Revision Date 3 | DATEFORMAT | YES | YES | [X] After RREL53 (C41). | Not mapped. | no |
| RREL56 | Revised Interest Rate Index | LIST | YES | YES | [X] Present without any revision date (C41). | Not mapped. | no |
| RREL57 | Revised Interest Rate Index Tenor | LIST | YES | YES | [X] Present without RREL56 (C41). | Not mapped. | no |
| RREL58 | Number Of Payments Before Securitisation | INTEGER-9999 | YES | NO | A count of payments, not months. [P] Against the payments possible between RREL23 and RREL7 at the RREL37 frequency (C26); not evaluated when RREL7 or the frequency is missing. | MILAN Number of Payments Before Securitisation. | no |
| RREL59 | Percentage Of Prepayments Allowed Per Year | PERCENTAGE | YES | YES | [E] Between 0 and 100 inclusive (validation rule). [P] One value on every record (P9). Zero is the prescribed entry where overpayments are not allowed without penalty (Q&A 1354), so it is never a placeholder finding. | Not mapped. | no |
| RREL60 | Prepayment Lock-Out End Date | DATEFORMAT | YES | YES | [X] Not before RREL23 (C42). | Not mapped. | no |
| RREL61 | Prepayment Fee | MONETARY | NO | YES | [P] Above RREL30 (C42). | Not mapped. | no |
| RREL62 | Prepayment Fee End Date | DATEFORMAT | YES | YES | [X] Not before RREL23 (C42). | Not mapped. | no |
| RREL63 | Prepayment Date | DATEFORMAT | YES | YES | [X] Not before RREL23, not after cut-off; against RREL64 (C42). | Not mapped. | no |
| RREL64 | Cumulative Prepayments | MONETARY | YES | YES | [P] Above RREL29 (C22). | Not mapped. | no |
| RREL65 | Date Of Restructuring | DATEFORMAT | YES | YES | Own validator, taking precedence over generic P5. The XML carries repeated date elements; a `,` or `;` inside one CSV cell is this application's adapter convention (stated as such, not an XSD literal). The audit keeps: the original cell; the separator found; every element, each classified as a valid date, a **recognised ND state** (confirmed in the package: the element repeats and each occurrence is a date, ND1, ND2, ND3, ND5 or a dated ND4) or an invalid fragment; and the date the pipeline selected. [E] Each **date** element is not later than the cut-off (validation rule). The date inside a dated ND4 element is an availability date: it is tested by P3e (after the cut-off) and never by the event-date rule. Three questions stay separate: schema validity of each element; semantic justification (an ND option beside a real date is allowed by the schema and reported as a question, "mixture not explained"); and adapter support (L: what the pipeline's split and latest-date selection did with the cell, observed and reported whatever the elements were). A cell of two valid dates passes. A cell with one valid date and one invalid fragment keeps the fragment's violation even though the pipeline drops the fragment and selects the valid date (L note). [X] Empty, invalid marker or ND5 on RNAR or RARR (C9a). A date on a performing loan is history (C9b). | The pipeline splits on `,` or `;`, keeps parseable dates and takes the latest; unparseable fragments disappear from its value. MILAN Restructured Loan, Restructuring Date. | no |
| RREL66 | Date Last In Arrears | DATEFORMAT | YES | YES | [X] After cut-off (C8); missing while RREL68 is above 0 (C7). | MILAN Months Current. | yes |
| RREL67 | Arrears Balance | MONETARY | NO | NO | [E] "If no arrears then enter 0"; not below 0 (P6). [X] Against RREL68 and RREL69 (C6). [P] Above RREL30 is possible (fees and capitalised amounts), reported as unusual. | MILAN Arrears Amount. | yes |
| RREL68 | Number Of Days In Arrears | INTEGER-9999 | NO | NO | [X] Against RREL67 and RREL69 (C6); above the days since RREL23. | MILAN Months In Arrears, Months Current. | yes |
| RREL69 | Account Status | LIST | NO | NO | [L] A code that is active only after trimming or upper-casing is excluded by the exact comparison in Stage 2 and is counted in section 8 as "active status not recognised". [X] Lifecycle rules (C9, C10, C27, C43). | Required column. Stage 2 filter; MILAN Account Status (OTHR reads "ND"), Restructured Loan. | yes |
| RREL70 | Reason for Default or Foreclosure | LIST | YES | YES | [E] Applicable when the exposure is in default under Article 178: statuses DFLT and DTCR (C43a). Not required for NDFT or DADB. | MILAN Reason for Default or Foreclosure. | no |
| RREL71 | Default Amount | MONETARY | NO | YES | Lifecycle rule, citing the Annex text and Q&A 1324 together (C10): kept unchanged when a defaulted exposure performs again; updated on a further default. For an exposure that never defaulted the Annex says enter 0 and Q&A 1324 says "in all other cases ND5 must be entered"; the two sources differ, so ND5 is compliant and 0 is an accepted alternative citing the Annex, and neither is a violation. [X] 0 or empty on a default status (C43b). | MILAN Default amount. | no |
| RREL72 | Default Date | DATEFORMAT | NO | YES | [X] Missing on a default status (C43b); after cut-off; before RREL23. Kept on a re-performing loan (Q&A 1324): a date on a performing loan is compliant history. | MILAN Default date. | no |
| RREL73 | Allocated Losses | MONETARY | NO | YES | Negative is prescribed for a gain on sale, and the workbook has no sign rule (removed in 1.3.1). Q&A 1348: ND5 where no recovery or work-out has begun. [P] Non-zero with no default history at all (C43c). | MILAN Allocated Losses. | no |
| RREL74 | Cumulative Recoveries | MONETARY | NO | YES | [E] May be negative (no sign rule). Q&A 1344: ND5 where no recovery process exists; 0 where one is or was under way with nothing recovered. [P] Above 0 with no default history (C43c). | MILAN Cumulative Recoveries. | no |
| RREL75 | Litigation | Y/N | NO | YES | Boolean spelling. Distribution only. | Not mapped. | no |
| RREL76 | Recourse | Y/N | YES | YES | Boolean spelling. | MILAN Recourse accepts `Y`/`N` only (#47). | no |
| RREL77 | Deposit Amount | MONETARY | NO | YES | [E] Not below 0 (P6). [X] Same currency as the exposure (C30). The definition's caps and Q&A 1343 depend on offset law and the reporter's savings convention, which the files do not hold, so above RREL30 is [P] only (C44). | MILAN Deposit Amount. | no |
| RREL78 | Insurance Or Investment Provider | ALPHANUM-1000 | YES | YES | Text rules only. | MILAN Additional Collateral Provider. | no |
| RREL79 | Original Lender Name | ALPHANUM-100 | YES | YES | [X] Pairing with RREL80, RREL81 (C45, Info). | Not mapped. | no |
| RREL80 | Original Lender Legal Entity Identifier | LEI | YES | YES | LEI rules; entity match not assessed. | Not mapped. | no |
| RREL81 | Original Lender Establishment Country | COUNTRYCODE_2 | YES | YES | ISO membership; `NA` is Namibia here, a value (the loader reads it as missing, P3g). Not a reference for residency. | Not mapped. | no |
| RREL82 | Originator Name | ALPHANUM-100 | NO | NO | Text rules. Several originators in one file are legitimate; the distinct names are listed (Info). | MILAN Originator Identifier (column 1). | yes (presence and format only) |
| RREL83 | Originator Legal Entity Identifier | LEI | NO | NO | LEI rules. [X] One LEI per originator name and the reverse (C46, Info). | Not mapped. | no |
| RREL84 | Originator Establishment Country | COUNTRYCODE_2 | NO | NO | ISO membership; `NA` is a value. Not a reference for residency. | Not mapped. | no |

#### Collateral file

"Property row" means RREC5 is CBLD, RBLD, IBLD or OTRE. "Established
non-property" means RREC5 is one of the vehicle, equipment, goods,
securities, other-financial-asset or guarantee codes (the pipeline's 19
excluded codes). OTHR and MIXD are **undetermined**: no property or
non-property rule is asserted on them, and the pipeline does not exclude
them. A guarantee backed by collateral should have been looked through to
that collateral (RREC5 definition), so GUAR means no look-through asset
was reported.

| Code | Field | FORMAT | ND1-4 | ND5 | Field-specific rules | Pipeline | Crit. |
|---|---|---|---|---|---|---|---|
| RREC1 | Unique Identifier | ALPHANUM-28 | NO | NO | As RREL1; equal to RREL1 (A11). | Dropped at Stage 1. | no |
| RREC2 | Underlying Exposure Identifier | ALPHANUM-1000 | NO | NO | [E] "This must match field RREL3" (F3a). [L] What the pipeline matched it to (F3b). | Required column. The join to the loan. | yes |
| RREC3 | Original Collateral Identifier | ALPHANUM-1000 | NO | NO | [X] Mapping to RREC4 unambiguous (E15). | Not in the Stage 1 minimum list, but it is a candidate in property-identifier selection (`original_id_col`), so its absence can fail Stage 3 and its values can become the property id. MILAN Additional data 26. | yes |
| RREC4 | New Collateral Identifier | ALPHANUM-1000 | NO | NO | [X] Unique with RREC2 (A8). | Required column. Property id; graph node; MILAN Property Identifier. | yes |
| RREC5 | Collateral Type | LIST | NO | NO | Classification into property, established non-property, undetermined (above). [P] GUAR while RREC23 says no guarantor (E18). | Required column. Stage 2 keeps rows whose code is not one of the 19 excluded codes. | yes |
| RREC6 | Geographic Region - Collateral | NUTS | YES | YES | [E] NUTS 3 or country plus `ZZZ` (P11); "physical collateral". Used by C28. | MILAN Property Postal, Region code or Nuts code (main property only). | no |
| RREC7 | Occupancy Type | LIST | YES | YES | [E] ND5 on established non-property; a value there is a breach (P3f / E17). ND5 on a property row is questioned (E17b). | MILAN Occupancy Type (all codes mapped). | no |
| RREC8 | Lien | INTEGER-9999 | YES | YES | "Highest lien position held by the originator in relation to the collateral", not the loan's own rank. [E] ND5 on established non-property collateral **other than guarantees**; for a guarantee the field is the guarantee holder's priority on the guarantor, possibly 1 (Q&A 1323(c)), so a value on a GUAR record is compliant. [P] 0 is not a position. | MILAN Additional data 14. | no |
| RREC9 | Property Type | LIST | NO | YES | [E] ND5 on established non-property; a property code there is a violation. [E/L] ND5 or empty on a property row of a reporting-active exposure: questioned use of ND5, and the row is excluded at Stage 2 (E2). On a final-inactive exposure ND5 is the prescribed value (Q&A 1296). | Required column. Stage 2 keeps rows with a non-missing value; MILAN Property Type (all codes mapped, OTHR to "10"). | yes |
| RREC10 | Energy Performance Certificate Value | LIST | YES | YES | [X] Pairing with RREC11 (E13). Q&A 1353 accepts ND5 here and in RREC11 where the information is unavailable, so ND5 is never questioned. | MILAN Energy Performance Certificate Value. | no |
| RREC11 | Energy Performance Certificate Provider Name | ALPHANUM-100 | YES | YES | Text rules; pairing (E13). | MILAN Energy Performance Certificate Provider Name. | no |
| RREC12 | Current Loan-To-Value | PERCENTAGE | YES | YES | [E] ND5 rule as RREC7. [E] 0 where the loan balance is negative; a 0 with a positive balance is a possible placeholder (P8). [P] `999.99999`. For non-first-lien loans the field is the combined LTV, so recomputation is qualified (E9). | MILAN Current LTV (divided by 100). | no |
| RREC13 | Current Valuation Amount | MONETARY | YES | YES | [E] For a guarantee: the amount guaranteed. [E] Pro-rated across the exposures sharing the collateral (Q&A 1323(d)). [L] Missing or not above 0 together with RREC17: excluded at Stage 2 (E3). [L] At most 10: valuation selection flips once to the other valuation and does not re-test it (E4). Supporting method and date (E5). Currency against RREC22 (C30). | Required column. Stage 2 filter (either valuation above 0); valuation selection; MILAN Property Value. | yes |
| RREC14 | Current Valuation Method | LIST | YES | NO | [X] Triplet (E5). | Valuation selection prefers FIEI/FOEI; MILAN Property Valuation Type (all codes mapped). | yes |
| RREC15 | Current Valuation Date | DATEFORMAT | YES | YES | [X] Not after cut-off; not before RREC19 (E6); triplet (E5). | Valuation selection; MILAN Property Valuation Date. | yes |
| RREC16 | Original Loan-To-Value | PERCENTAGE | YES | YES | As RREC12 (ND5 rule; combined LTV for non-first liens). | MILAN Original LTV. | no |
| RREC17 | Original Valuation Amount | MONETARY | YES | NO | "Used when the underlying exposure was originated": specific to the exposure, so not compared across loans sharing the property. [P] Equal to RREC13 on most records (E7, "unusual", not proof of copying). | Stage 2 filter; valuation selection. | yes |
| RREC18 | Original Valuation Method | LIST | YES | NO | [X] Triplet (E5). | Valuation selection. | yes |
| RREC19 | Original Valuation Date | DATEFORMAT | YES | NO | [X] Not after cut-off, not after RREC15 (E6). [P] Far from the loan's RREL23 (E6, Info). | Valuation selection. | yes |
| RREC20 | Date Of Sale | DATEFORMAT | YES | YES | [X] Not after cut-off. A past sale on a performing loan is history (E12, Info). | Not mapped. | no |
| RREC21 | Sale Price | MONETARY | NO | YES | "Price achieved on sale ... in case of foreclosure". [X] A price without RREC20 (E12). [P] `0` on every record without any sale date: possible placeholder for ND5; zero proceeds and unsold are different states (P8). | MILAN Sale Price. | no |
| RREC22 | Collateral Currency | CURRENCYCODE_3 | NO | YES | [E] "The currency in which RREC13 is denominated": equal to the RREC13 companion (C30). | Not mapped. | no |
| RREC23 | Guarantor Type | LIST | YES | NO | [P] Against GUAR rows (E18). | MILAN MIG Provider reads "No Guarantor" when missing or unrecognised; Type of Guarantee Provider reads "ND" when missing. Two different consequences from one field. | no |

### 6.4 The inventories as report artefacts

**Expected fields** (107 rows): file; code; field name; FORMAT; ND1-4 and
ND5 allowance; present; header as found and position; record counts by
value state (nine columns); the same on the retained active pool; ND rule
results (P3a to P3g, counts); format compliant / XML-form / non-compliant
counts with up to five distinct examples; domain, length, reference-list
results; possible placeholders with counts; distinct values and, for LIST
fields, the code distribution; whitespace count; field-specific rules
that fired with counts; pipeline use and lineage class (9.3); outcome
per rule group (the coverage counts of 3.4: applicable, evaluated,
failed, other result kinds, not evaluable); worst
severity by basis (E, L, X, A, P) in separate columns; critical.

**Actual columns** (one row per physical column): as 5.3.

**O1. Optional ESMA elements.** Between the 107 expected fields and an
ordinary extra column sits a third kind: an element the XSD offers and
the RTS does not list. There are two, current and original gross annual
rental income for buy-to-let (4.2 item 17). Each has a catalogue entry
marked optional. Absent: no finding, and it is not in the P1 inventory.
Supplied: validated as an amount (P5, not below 0, currency evidence),
with ND5 allowed and ND1 to ND4 not (P3a), and shown in the
actual-column inventory as "recognised optional ESMA element,
assessed". The header names that recognise such a column are this
application's convention. If a later package adds an optional element
whose rules are not yet transcribed, the inventory says "recognised
optional ESMA element; compliance not assessed" and it is a coverage
limitation, never a pass.

**As implemented.** The two optional elements are recognised in a
collateral file by the cleaned header names
`current_gross_annual_rental_income` and
`original_gross_annual_rental_income` (`DQ_OPTIONAL_ELEMENT_HEADERS`),
each with `<name>_currency` as its companion. In the expected-field
inventory: the value states on the retained active pool are given for
loan fields from the population label, and for collateral fields from
the link (5.2): a collateral record belongs to the retained pool when a
raw link under either convention leads to a retained loan. "Pipeline use and lineage class" is empty and says it arrives
with P10. In the actual-column inventory, the sample values are up to
five most frequent distinct values of the user's own file, each cut at
100 characters; they are written to the summary beside the workbook and
never logged. Record numbers and column positions are 1-based.

---

## 7. Part 2: consistency checks across columns, records and files

Each check states its basis (3.2) and the records it applies to. All run
on every applicable record of the raw files; "Counts" says whether a
finding on the retained active pool counts towards rejection rules 2 and
4 (critical). A check is not evaluable on a record when a value it needs
as an input to a comparison is missing or unparseable, and says so; a
check whose tested condition is the missing value itself (an omission
rule) stays evaluable (3.4). Any check that compares, adds or divides
two monetary amounts needs both currencies stated and equal (6.2);
otherwise it is not evaluable for that record and says so, and no
conversion is attempted. Ids are kept from the
earlier documents where the check survives; retired ids are in Appendix A.

### 7.1 Family A: the files

| Id | Check | Basis | Severity | Counts |
|---|---|---|---|---|
| A1 | No header, no data records (L). Names colliding after cleaning (L, an existing processing failure). The same header text twice: today accepted with the second column renamed; a Stop here is a **proposed acceptance policy** (A), assumption d in section 15. | L; A | Stop | yes |
| A2 | A column the pipeline requires is absent. Loans: RREL2, RREL3, RREL5, RREL23, RREL30, RREL69 and `pool_cutoff_date`. Collateral: RREC2, RREC4, RREC5, RREC9, RREC13 (`REQUIRED_*_COLUMNS`). These are the pipeline's minimum and are not the ESMA requirement (P1 covers all 107), nor a full list of what later stages read: a missing valuation or lien column can fail a later stage, which section 9.2 reports as that stage's data fault. | L | Stop | yes |
| A3 | Header resolution: how each column was recognised; unmapped columns (5.3) | L (context) | Info | no |
| A4 | Expected fields absent (one P1 finding per field) | E | Warn | no |
| A5 | A value the loader cannot convert in a numeric field, on any record, including a dated ND4 (P3g). The loader raises on the first; the audit lists all. | L | Stop | yes |
| A6 | Bytes that are not valid UTF-8 and were repaired, per field: evidence about ingestion, not in itself a reporting breach; the field's own format rule decides whether the repaired value complies | L | Warn | no |
| A7 | The same RREL3 (or the same RREL2) on more than one loan record; whole-record duplicates separately | X | Warn | yes |
| A8 | The same (RREC2, RREC4) pair on more than one collateral record; whole-record duplicates separately | X | Warn | no |
| A9 | More than one cut-off date across RREL6, `Pool_Cutoff_Date` and the collateral file, or the cut-off missing on some records | A | Stop | yes |
| A10 | (a) Cut-off after the assessment date: a violation with a regulatory source (workbook: RREL6 "must be in the past"), the single owner of that defect (P6(e) points here; one defect key). The assessment date is the calendar date of the run's start in UTC, recorded in provenance, and the comparison is by calendar date: a cut-off of tomorrow or later is the violation; a cut-off equal to the assessment date is a question, because the source's wording does not settle whether the same day is "in the past"; yesterday or earlier passes. (b) Cut-off before every origination date: unusual. | (a) E; (b) P | Warn | no |
| A11 | RREL1 not constant, RREC1 not constant, or the two differ | X | Warn | no |
| A12 | A value in one loan-identifier column equals a different record's value in the other (#45) | L | Stop until #45 merges, then Warn | yes |
| A13 | (a) An identifier with leading or trailing whitespace (#46). (b) Two different raw identifiers that become equal after trimming. | L | (a) Stop until #46 merges, then Warn; (b) Warn, and identity is flagged ambiguous | yes |
| A14 | Record counts of each file | L (context) | Info | no |
| A15 | One field supplied under two headers (table in 5.3). Not a cited regulatory breach: a CSV representation policy (A) for the Warn, an actual processing failure (L) where the names collide, and a policy Stop (A) where the values differ. | A / L | Warn or Stop | no / yes |
| A16 | A file that cannot be split into records (3.4) | L | Stop | yes |
| A17 | **Currency compatibility of the amounts the pipeline combines** (8.5): among retained loans and their selected collateral, more than one currency, or an invalid one, within any set of amounts that the pipeline adds, subtracts, divides or allocates together. By the owner's decision (15.1, decision 3) the pool is processed and this is a prominent warning, not a Stop. | L | Warn | no |

**Settled readings of family A** (as implemented). Every rule but A17
is evaluated from the two files alone, while the file's text is in hand.
Where one id of the table holds cases with different bases or
severities, the catalogue splits it with a letter: A1(a) no header or no
record, A1(b) one pipeline name for two columns, A1(c) a repeated header
text; A7(a) and A8(a) the duplicate identifier, A7(b) and A8(b) the
whole-record duplicate; A10(a) and (b); A13(a) and (b); A15(a) equal
values and A15(b) differing values.

- **A1 and A16.** A1(a) owns a file with no header and a header with no
  record. A16 owns a file that cannot be split into records (a record
  longer than the header, an unclosed quote, line ends the parsers read
  differently) and gives it its rows: one for the file and one for every
  over-long record.
- **A1(b), A1(c) and A15** are the three cases of 5.3.
- **A2** shares the defect key of P1 on the same field. Both rows are
  shown: P1 is the reporting requirement, A2 the processing one.
- **A3, A4 and A14** are reported through what already exists and say
  so, as P2 and P4(c) do: A3 through the actual-column inventory, with
  one context row per unmapped column; A4 is P1; A14 is the summary's
  file block.
- **A5 is the loader's test, not the format rule.** The loader casts
  every column whose pipeline name is a numeric field's, in either file,
  and accepts what a float cast accepts. A5 applies the same test to the
  raw column (null tokens removed, whitespace trimmed, a float cast), so
  it lists exactly the cells that make the loader stop, including in a
  column the audit calls unmapped (a loans column named `lien`). `1e3`
  converts and is a P5 violation with no A5 row; `abc` is both; a dated
  ND4 in a numeric field is A5's row alone. A5 and P5 keep two defect
  keys: one is a reporting breach in any pipeline, the other this
  loader's limit (8.4).
- **A6** is context on each repaired cell (and on a repaired header).
  The detail file gives the repaired text and the original bytes.
- **A7 and A8** ignore a cell that is not an identifier: an empty cell
  and an ND token, which the ND rules report. A whole-record duplicate
  is a record whose every cell equals another record's.
- **A9** has one row for the pool, listing every distinct cut-off date
  with its source and record count, and one row for each record whose
  cut-off is missing, or conflicts within the record, or differs from
  the date most records of the loans file hold (the later date on a
  tie).
- **A10(a)** compares each distinct cut-off date with the assessment
  date of the provenance: after it, a violation; equal to it, a
  question; before it, nothing. **A10(b)** is a question when the single
  cut-off is before the earliest valid RREL23.
- **A11** is tested in each file and across the two; a file with no
  identifier at all takes no part in the comparison.
- **A12** uses the definition of the identity-confidence label (8.1): a
  value that is a different record's value in the other identifier
  column. A record repeated whole therefore collides with its copy.
- **A13** covers RREL2 to RREL5 and RREC2 to RREC4; (b) is tested within
  each column.
- **Severity words.** A2, A5, A16, A1(a) and A1(b) have basis L: "the
  pipeline cannot process this". A9, A1(c) and A15(b) have basis A:
  "would stop under the acceptance policy". A12 and A13(a) are Stop
  while `DQ_A12_IS_STOP` and `DQ_A13_IS_STOP` are true, worded "the
  pipeline halts or mismatches on this today (issue #45 / #46)". No row
  says the run was stopped; only the pipeline's own failure says that.
- **A17** is evaluated after the run, among the loans Stage 2 retained
  (the audit's retained loans when Stage 2 did not complete). RREL30 is
  compared across the pool: one pool-level row with every currency and
  its record count, and one row for each record that is not in the most
  frequent valid currency or holds a code that is not a valid currency.
  RREL30 with RREL31 to RREL33, and RREL16 with RREL20, are compared on
  each loan. The fourth set is the valuation Stage 7 actually selected
  for each collateral record of the loan (the RREC13 companion and
  RREC22 for a current valuation; the RREC17 companion for an original
  one), with the loan's RREL29 and RREL30, and across the pool; it is
  "not evaluable: Stage 7 did not complete" when the pipeline raised
  before it. A currency that is not stated cannot be compared: such
  records are counted as not evaluable, apart, and P13 reports them.

### 7.2 Family C: within a loan record

"Active" means the status is one of PERF, ARRE, RARR, RNAR; "default
status" is DFLT, NDFT, DTCR or DADB; "repurchase status" is REBR, REDF,
RERE, RESS or REOT.

| Id | Check | Basis | Applies to | Severity | Counts |
|---|---|---|---|---|---|
| C1 | RREL30 above RREL29: unusual; capitalised arrears or further advances can explain it | P | all | Warn | no |
| C2 | RREL41 above RREL30 | X | all | Warn | no |
| C3 | RREL23 after cut-off (X with a regulatory source: counts in compliance). RREL24 not after RREL23 (X). RREL24 on or before cut-off on an active loan; more than 50 years after cut-off (P) | X / P | all / active | Warn | yes for the X parts |
| C4 | RREL7 before RREL23 or after cut-off | X | all | Warn | no |
| C5 | RREL25 differs from the months between RREL23 and RREL24 by more than 3 | P | all | Warn | never (decision) |
| C6 | (a) ARRE or RARR with both RREL68 = 0 and RREL67 = 0 (active statuses). (b) PERF or RNAR with RREL68 above 0 (active statuses). (c) RREL68 above the days since RREL23 (every record). (d) RREL67 above RREL30: unusual only (every record). | X; (d) P | per part | Warn | (a) to (c) yes |
| C7 | RREL68 above 0 with RREL66 empty or an invalid marker: a date that must exist is omitted (X, violation). RREL66 a permitted ND1 to ND4: unavailable (context). ND5 with days above 0: question (P3f). Pipeline note (L): MILAN Months Current reads "0" whenever days in arrears are above 0, whatever the date, so the output is unaffected. | X / context / L | all | Warn; Info | X part yes |
| C8 | RREL66 after cut-off | X | all | Warn | yes |
| C9 | (a) RNAR or RARR with RREL65 **empty, an invalid marker or ND5** (violation: a restructured status establishes applicability). RREL65 a permitted ND1 to ND4: unavailable, context. (b) RREL65 on a status that is not restructured: history, context | X; context | all | Warn; Info | no |
| C10 | (a1) RREL9 a date on an active status (RREL9 is legitimate on RDMD and on a default status whose recovery has ended): violation. (a2) RREL8 a date on a status that is not a repurchase status: a **question**, never a violation, as in the RREL8 row; it does not count. (b) RREL72 and RREL71 on a performing loan: compliant history under Q&A 1324, shown as context. (c) RREL71 above 0 while RREL72 is ND5 or empty (an amount with no default event), on any status. | X; (a2) P; (b) context; (c) X | all | Warn; Info; Warn | (a1) yes; (a2) no; (c) no |
| C11 | RREL43 zero, negative or above 20: unusual | P | active | Warn | no |
| C12 | Rate-type matrix. (a) FLIF, FINX, FLFL, CAPP, FLCA without RREL44. (b) FXRL with RREL44 or RREL46 populated. (c) FXPR or FLCF with RREL51 empty, an invalid marker or ND5: Q&A 1334 puts the reset or switch date there, and Q&A 1295 says the type becomes FLIF once the switch has happened, so this is a violation (E, Q&A); a permitted ND1 to ND4 is context. No rule requires RREL51 on an ordinary floating loan. DISC, SWIC, OBLS, MODE, OTHR: no rule. | X; (c) P | all | Warn | (a) yes |
| C13 | RREL51 on or before cut-off, or after RREL24. Basis: the field definition ("date interest rate next changes") and Q&A 1334; not a workbook rule (4.2 item 6). A permitted ND is never a finding here. | X | all | Warn | no |
| C14 | RREL44 LIBO or EURI without RREL45: MILAN Base Index reads "ND" | L | all | Warn | no |
| C15 | (a) RREL51 present while RREL50 is empty or an invalid marker (X); RREL50 a permitted ND while RREL51 is a date is not a finding: a re-fix of a fixed rate need not have a margin. RREL50 a value while RREL51 is ND5: questioned (P). (b) RREL46 above RREL43: possible with a negative index | X / P; P | all | Warn; Info | no |
| C16 | RREL39 zero on an amortising active loan; RREL39 above RREL30 | P | active | Warn | no |
| C17 | (a) RREL36 before cut-off: the report asks whether ND5 was intended; a question, not an established breach. (b) RREL36 after RREL24 | P; X | all | Info; Warn | no |
| C18 | RREL31 missing: External Prior Ranks CB reads "ND". RREL32 missing: Pari Passu reads "0", i.e. treated as none | L | active | Info | no |
| C19 | RREC8 above 1 on a loan's only property row while RREL31 is 0: shown for context. The reverse implication is dropped: RREC8 is the originator's highest lien on the collateral, not the loan's rank. | P | active | Info | no |
| C20 | (a) RREL16 below 1,000 or above 10,000,000: unusual. (b) RREL17 CORP with RREL13 not NOEM, or the reverse | P | all | Warn | no |
| C21 | Pool-level: median RREL40 below `DQ_DTI_SCALE_SUSPECT_MEDIAN`: the column may be a ratio or a payment-to-income measure. Reported as a suspicion; values are not rescaled and no row-level bound is derived from the median. | P | pool | Warn | no |
| C22 | RREL64 above RREL29 | P | all | Warn | no |
| C23 | RREL34 other than 100 | P | all | Info | no |
| C24 | RREL40 negative. Q&A 1322 makes RREL30 the numerator and a negative RREL30 is valid, so the sign alone proves nothing. With RREL30 below 0: context (the ratio follows the balance). With RREL30 above 0: a question, since the total income behind the ratio would have to be negative and "other income" is not in the files. No violation is asserted. | P | all | Warn; context | no |
| C25 | RREL31 equal to the loan's own RREL30. Not a mathematical impossibility and not prohibited by the taxonomy; a high share is a strong reason to query the export. Result kind **policy trigger**: it counts towards rejection because the owner approved that acceptance condition, is overridable, and is never a compliance violation. | A | active | Warn | yes |
| C26 | RREL58 above the number of payments possible between RREL23 and RREL7 at the RREL37 frequency, plus a tolerance of 3. Not evaluated without RREL7 or a frequency; the cut-off is never substituted for RREL7. No direct comparison with RREL25 (payments against months). | P | all | Warn | no |
| C27 | RDMD with RREL9 empty, an invalid marker or ND5: violation (Q&A 1341). A repurchase status with RREL8 empty or an invalid marker: violation; with RREL8 = ND5: a question (3.3, provisional). RREL8 or RREL9 before RREL23: violation. RREL8 or RREL9 after cut-off is P6(b)'s defect, shared key, counted once. | X; E | all | Warn | no |
| C28 | Residency. Assessable only when RREL10 and RREL11 are valid, the loan has at least one property collateral record with a valid RREC6, and all its property records share one country; otherwise not evaluable, with the reason (several collateral countries, missing location). The files carry no separate location of the exposure, so the collateral country stands for "the country in which the collateral and underlying exposure reside"; that assumption is printed with the finding. Then: resident while the RREL11 country differs from the collateral country, or non-resident while they are equal. A cross-border loan is not a finding. | X (qualified) | all | Warn | no |
| C29 | Retired (it reversed the meaning of RREL14) | | | | |
| C30 | Currency of related fields: RREL28 against the RREL29 and RREL30 companions; RREL18 against the RREL16 companion; RREC22 against the RREC13 companion; the RREL77 currency against the exposure's currency (its definition: "use the same currency denomination as that used for this underlying exposure"). Different currencies between income and loan, or collateral and loan, are reported as information (FX exposure), never as a format failure. | X; Info | all | Warn; Info | no |
| C31 | RREL21 holds a verification code while RREL20 is ND5; RREL20 above 0 while RREL21 is ND5. A code with RREL20 = 0 is a verified zero income and is not a finding. Permitted ND1 to ND4 on either side: unavailable, Info. | X | all | Warn | no |
| C32 | One RREL23 date on more than half of the loans: unusual | P | pool | Info | no |
| C33 | RREL29 equal to RREL30 on more than 90% of seasoned amortising loans: unusual | P | pool | Info | no |
| C34 | RREL33 above 0 and below RREL30: a limit below the drawn balance. MILAN Flexible Loan Amount reads "0" (floored). | P | all | Warn | no |
| C35 | BLLT with RREL41 empty, an invalid marker or ND5 (violation); RREL41 a permitted ND1 to ND4: unavailable, context. RREL41 above 0 on FRXX: question | X; P | all | Warn; Info | no |
| C36 | RREL37 differs from RREL38 | P | all | Info | no |
| C37 | RREL39 outside one third to three times an annuity estimate on monthly amortising loans | P | active | Info | no |
| C38 | RREL47 a value on FXRL or FLCF, where Q&A 1334 prescribes ND5; RREL47 empty, an invalid marker or ND5 on FXPR, where it prescribes the number of months | E (Q&A) | all | Warn | no |
| C39 | RREL45 present while RREL44 is missing | X | all | Info | no |
| C40 | RREL48 below RREL49; RREL43 above RREL48 or below RREL49; CAPP or FLCA without RREL48; FLFL or FLCA without RREL49 | X | all | Warn | no |
| C41 | Revision dates not ascending; RREL56 or RREL57 a value while every revision date is ND5; RREL57 a value while RREL56 is ND5. A revision date without a numeric margin is not a finding (as C15). Permitted ND1 to ND4 never create a finding here. | X | all | Warn | no |
| C42 | RREL60 or RREL62 before RREL23; RREL63 before RREL23 or after cut-off; RREL63 present while RREL64 is 0; RREL61 above RREL30 (P) | X / P | all | Warn | no |
| C43 | Default fields, by lifecycle branch. On a **final-inactive (presumed)** record the Q&A 1296 pattern applies and ND5 in these fields is not a finding. On every other record with a default status, including "default, recoveries not determinable", the submitted status itself requires the default fields (Q&A 1324 is status-based), so the rules below give violations, not questions: DFLT with RREL9 = ND5, a valid RREL72 and RREL71 = ND5 is a (b) violation. (a) DFLT or DTCR (Article 178 default) with RREL70 empty, an invalid marker or ND5: violation; a permitted ND1 to ND4: context; RREL70 on other statuses: context. (b) A default status with RREL72 empty, an invalid marker or ND5; or with RREL71 empty, an invalid marker, **ND5** or 0 (Q&A 1324: amount and date are completed for a defaulted exposure; ND5 is allowed for the field in general, so P3b cannot catch this, and C43 owns it with one defect key shared with P3f). (c) RREL73 or RREL74 non-zero on a loan with no default status and no RREL72: question. RREL72 after cut-off or before RREL23: violation. | E; X; P | all | Warn | no |
| C44 | RREL77. Its definition caps the amount at the lower of the deposit and the maximum potentially off-settable amount at obligor level, and Q&A 1343 ties "potentially off-settable" to the national legal framework; neither is in the two files, so no cap breach is asserted. RREL77 above RREL30 in the same currency: unusual, worded "a net-balance savings convention (Q&A 1343) or an allocation across the obligor's loans can explain this"; the reporting convention is never inferred from the excess. Different or unknown currencies: not evaluable for the comparison, and the currency difference itself is C30's. | P | all | Warn | no |
| C45 | Original lender fields. A lender name (RREL79) with RREL80 **empty or an invalid marker**, or the reverse: omission, violation. Either side a permitted ND1 to ND4 while the other is a value: unavailable, context. ND5 on one side while the other is a value: question. Both ND: nothing. | X | all | Info | no |
| C46 | One originator name with several LEIs or countries, or one LEI with several names. Several originators in a file are not a finding. | X | pool | Info | no |
| C47 | DTI recomputation, a plausibility note only. One loan record for an obligor in these files does not establish the obligor's whole debt or how income was allocated (Q&A 1322 pro-rates income across loan parts), so the note never claims an upper bound. Evaluated only for an obligor with exactly one loan record, all inputs present, positive and in one currency: reported RREL40 beside 100 × RREL30 / (RREL16 + RREL20), flagged when they differ by more than a factor of two. Obligors with several records: no comparison (repeated income reports would be double-counted and may come from different underwriting dates). | P (qualified) | all | Info | no |

### 7.3 Family D: a borrower's records

| Id | Check | Basis | Severity | Counts |
|---|---|---|---|---|
| D1 | The same borrower with different RREL13, RREL10 or RREL11 on loans originated within 12 months of each other. Income, income type and verification are not compared: they belong to each loan's underwriting date. | P | Info | no |
| D2 | Identifier mapping: one RREL4 with several RREL5 values, or one RREL5 with several RREL4 values (ambiguous mapping). A record whose RREL4 and RREL5 differ is the expected sign of an identifier change and is only counted. | X | Warn (ambiguous); Info (count) | no |
| D3 | Borrowers with more than 20 active loans | P | Warn | no |

### 7.4 Family E: collateral records, and collateral against the loan

| Id | Check | Basis | Applies to | Severity | Counts |
|---|---|---|---|---|---|
| E1 | RREC5 is one of the 19 excluded codes: the record is not used by the pipeline. Listed with its loan. Not a compliance finding. | L | all | Info | no (F2 carries the exclusion) |
| E2 | Property row with RREC9 empty or ND5. Regulatory part, by lifecycle: on a reporting-active (presumed) exposure a question about the use of ND5; on a final-inactive (presumed) exposure the prescribed pattern (Q&A 1296), no finding; where lifecycle is not determinable, a limitation. Processing part, independent of lifecycle: Stage 2 excludes the record (L, observed), result kind violation of a processing prerequisite (3.2). One check, one defect. Only the processing part counts, and only for a retained loan left with no usable record (3.2, decision 7); on a surviving loan it is a Warn that does not count. | E / L | property rows | Warn | yes (the L exclusion, on retained loans) |
| E3 | Both RREC13 and RREC17 missing or not above 0: Stage 2 excludes the record. Result kind: violation of a processing prerequisite (3.2), counted as in E2: only for a retained loan left with no usable record. On a final-inactive (presumed) record the zero RREC13 is the prescribed value and the exclusion is context. | L | property rows | Warn | yes |
| E4 | Valuation selection, from the Stage 7 result: the valuation preferred first; whether it was missing or at most 10 and selection flipped to the other; the alternative selected; and whether that alternative is itself at most 10 (it is not re-tested: a current value of 5 with an original of 8 selects 8). The last case is a Warn. | L | property rows | Info; Warn | no |
| E5 | Valuation support, on reporting-active exposures. (a) A valuation amount with its method or date **empty or an invalid marker**: omitted supporting information (X, violation). (b) Method or date a permitted ND1 to ND4: unavailable (context). (c) An amount of 0 with a method or date present: question. (d) Whether the valuation the pipeline selected for a property has a method and a date, since MILAN Property Valuation Type and Date read "ND" without them (L, from the Stage 7 result). Counting (d) towards rejection would be an acceptance-policy choice (question 4). | X / context / P / L | property rows | Warn; Info | (a) yes; (d) per question 4 |
| E6 | RREC15 before RREC19; either after cut-off. RREC19 far from the loan's RREL23: context | X; P | property rows | Warn; Info | no |
| E7 | RREC13 equal to RREC17 on more than 90% of property rows while the dates differ: unusual; does not prove copying | P | pool | Warn | no |
| E8 | RREC13 to RREC17 ratio outside 0.3 to 5: unusual. Computed only when both amounts are above 0 and both currencies are stated and equal; with different or unknown currencies the ratio is not evaluable and the row says why (no conversion is attempted, and A17 is not involved unless the valuation was selected). On a final-inactive (presumed) record RREC13 = 0 is the prescribed final value while RREC17 has no ND5 option and keeps its amount: no ratio and no warning, one context row ("current valuation reported as 0 on a final submission"). | P | property rows | Warn; context | no |
| E9 | Reported LTV against the file's own figures. A comparison is computed only when the denominator is present, finite and above 0; RREC13 of 0, or RREC13 − RREL31 of 0 or below, is reported as its own condition ("no positive value to compare against"), never as a tolerance failure. **Negative balance first:** where RREL30 is below 0 the definition of RREC12 prescribes 0, so the expected value is 0 and no ratio is recomputed (PERF, RREL30 = -1,000, RREC13 = 10,000, RREC12 = 0 is correct; a non-zero RREC12 there is the contradiction). All comparisons stay in the reported percentage units. On a final-inactive (presumed) record a zero RREC13, RREC12 or RREL30 is the expected final value: the comparison is not applicable there and the row is context, not a condition to explain. **Current** (RREC12 against RREL30 / RREC13): one loan with one property record, one currency, RREL31 = 0 and RREL32 = 0, tolerance 5 points. With RREL31 above 0, Q&A 1322 deducts the prior lien from the property value; comparator RREL30 / (RREC13 − RREL31), labelled qualified. **Original** (RREC16 against RREL29 / RREC17): a plausibility note (Info) under the same single-record conditions, never an asserted contradiction. Shared collateral, mixed currencies, missing inputs: not evaluable. Applies to inactive records too; never counts. | X (qualified); P | all | Warn; Info | no |
| E10 | The same RREC4 on several records. (a) Physical attributes differ (RREC6, RREC9) among records of reporting-active exposures: violation (X). (b) Current valuation method or date differ: sharing an identifier does not establish that every record reports the same valuation event, and a final-inactive record keeps old values (Q&A 1296). So (b) is a **question**, raised only among reporting-active records with the same cut-off; not critical. (c) Current valuation amounts: Q&A 1323(d) prescribes pro-rating, but equal amounts do not show that pro-rating was omitted (two equal shares of 400,000 are 200,000 each) and the files carry no unallocated value. The report states facts only: the amounts are equal or differ; the loans' balances; the convention the pipeline detected; and whether the allocation basis is established, uncertain or unavailable. No conclusion about pro-rating is asserted unless an independent full value exists in the data. Original valuation fields are exposure-specific and not compared. | X; question; context | property rows | (a) Warn; (b), (c) Info | no |
| E11 | Occupancy and purpose mix (buy-to-let indicators) | P | pool | Info | no |
| E12 | RREC21 above 0 without RREC20. A sale date on a performing loan's collateral: history, context | X; Info | all | Warn; Info | no |
| E13 | EPC fields. RREC10 a value with RREC11 **empty or an invalid marker**, or the reverse: omission, violation. A provider with the EPC value reported as a permitted ND1 to ND4 (or the reverse): unavailable, context. ND5 on one side with a value on the other: question. | X | property rows | Info | no |
| E15 | One RREC4 with several RREC3 values, or the reverse | X | all | Info | no |
| E16 | Properties referenced by several loans; loans with several properties | Info | all | Info | no |
| E17 | (a) Established non-property record with a value other than ND5 in RREC7, RREC9, RREC12 or RREC16; and in RREC8 except on GUAR records (Q&A 1323(c)). (b) Property row of a reporting-active exposure with ND5 in RREC7, RREC8, RREC12 or RREC16: question. On final-inactive exposures ND5 in the ND5-eligible fields and 0 in the current numeric ones is the prescribed pattern (Q&A 1296). OTHR and MIXD records, and records whose lifecycle is not established: limitation. | E | all | Warn; Info | no |
| E18 | GUAR record while RREC23 is NGUA; RREC23 a guarantor code on a loan with no GUAR record: context | P | all | Info | no |
| E19 | Every valuation date of a loan's collateral more than 12 months before RREL23: context | P | all | Info | no |

### 7.5 Family F: the relationship between the files

| Id | Check | Basis | Severity | Counts |
|---|---|---|---|---|
| F1 | Which loan-identifier column Stage 3 selected, the three gates' figures and the reason (from the stage's own result) | L | Info | no |
| F2 | Loans in the active pool that are excluded after Stage 2, by exclusion point (section 8.2); the balance feeds rule 3. Separately, a loan record with **no collateral record at all** is a compliance violation (E): ESMA refused any derogation from the requirement that every Annex 2 exposure has an associated collateral item, and a sold property keeps being reported with RREC20 and RREC21 (Q&A 1377, 1357). The violation does not count by itself; the loss it causes is already in rule 3. | L; E | Warn | rule 3 |
| F3 | (a) RREC2 that matches no RREL3 (the prescribed relationship). (b) RREC2 unmatched under the column the pipeline selected: violation of a processing prerequisite (3.2), counted only for a retained loan left with no usable record. (c) RREC2 that matches RREL2 only: processable, not the prescribed relationship. | (a) E; (b), (c) L | Warn | (b) yes |
| F4 | Collateral records of loans that are not in the active pool (count) | L (context) | Info | no |
| F5 | Loan records by population (section 8.1) | L (context) | Info | no |
| F6 | A loan with more than 50 properties | P | Warn | no |
| F7 | Groups larger than 20 loans or 20 properties | P | Warn | no |
| F8 | A group with zero total balance (equal split applied) | L | Warn | no |
| F9 | Aggregation convention detected, the evidence, and whether the fallback was used (from the stage's own result) | L | Info / Warn | no |
| F10 | Loans without any collateral record; collateral records per loan | X | Warn | via F2 |

### 7.6 Family G: pool level

These describe the pool. High LTV, concentration and unusual rates can be
credit risk, not poor data; the wording says so.

| Id | Check | Basis | Severity | Counts |
|---|---|---|---|---|
| G1 | Loans with computed current LTV above 1.5 or below 0.05; share above 1.0 | P | Warn | never (decision) |
| G2 | Weighted average rate, LTV and seasoning outside wide bands | P | Warn | no |
| G3 | Largest borrower above 5% of balance; top 20 above 50% | P | Info | no |
| G4 | Arrears share; arrears dates present but no loan in arrears | P | Info | no |
| G6 | Codes present in the pool that the MILAN code maps turn into "ND", read from the actual maps: today RREL35 OTHR, RREL69 OTHR, and index and tenor pairs without a Base Index entry. Most OTHR codes have a mapping and are not listed. | L | Warn | no |
| G7 | External Prior Ranks CB total above the pool's active balance | P | Warn | never (decision) |
| L1 | Consistency with the claimed lifecycle, on final-inactive (presumed) records only, by the four field categories of 3.3: a current numeric field that is not 0, or a value in a remaining ND5-eligible field. Never raised for identifiers, cut-off, status, a valid RREL9 under Q&A 1341, or preserved historical fields. Cites Q&A 1296. | E (question) | Warn | no |
| G8 | Population reconciliation (section 8.2): every difference between the audit's populations and the pipeline's own counts is explained by a listed exclusion. An unexplained difference is a technical fault, not a data finding, shown as "reconciliation incomplete" with both figures. Consequence: every calculation that uses the disputed population (the denominators and shares of rules 2 to 4, the waterfall lines concerned) is **unavailable**; the policy outcome is `unavailable` (10.1) and follows the technical contract, so there is no ordinary eligible release and no override; an outcome computed earlier from the disputed figures is not kept. A Stop that rests on evidence the difference does not touch still blocks. Finalised report files stay accessible, and facts the difference does not touch (per-column results, compliance violations, lineage) are still shown. | L (limitation) | technical | no; makes the outcome unavailable |

**G8, as implemented.** Two reconciliation points: line 5 of the
waterfall against the loans `Stage2Output` kept (the same records, not
only the same count), and line 7 against line 5 less line 6 (the records
of the output are exactly the retained loans not lost at line 6). A
difference is a row of kind limitation for the pool, with both figures,
and the waterfall line says "reconciliation incomplete". A point that
cannot be tested because the stage did not complete is counted as not
evaluable. The other rules of family G are not implemented yet.

---

## 8. Populations, exclusions and the verdict arithmetic

### 8.1 Populations

Processing retention and identity confidence are separate attributes; a
record can be retained and have an ambiguous identifier at once. The
table below is **retention only** and mirrors Stage 2 (status and
balance, nothing else), so it reconciles with the stage's output. Each
loan record is in exactly one:

| Population | Definition |
|---|---|
| Active, retained | status exactly one of the four active codes and balance above 0 (Stage 2 keeps it) |
| Active status, zero balance | excluded at Stage 2 |
| Active status, negative balance | excluded at Stage 2 |
| Active status, balance missing | excluded at Stage 2 |
| Active status, balance unparseable | the loader fails (A5) |
| Active status not recognised | the status equals an active code only after trimming or upper-casing (` PERF`, `perf`): excluded by the exact comparison, and listed because it was probably meant to be active. **Takes precedence over "Eligibility unknown"**: classification is tested in the order of this table. |
| Inactive | any other valid status code |
| Inactive status not recognised | the status equals a non-active list code only after trimming or upper-casing (`rdmd`, ` RDMD `): processed as not active (which is also what the exact comparison does), with the separate format or convention finding on the cell |
| Eligibility unknown | status missing, or not a list code even after trimming and upper-casing |

Identity confidence is a second label on every record: unique; duplicate
in a candidate identifier column that the pipeline did not select
(harmless to the selected identity); duplicate or colliding in the
selected column (ambiguous). Every count on the report says what it
counts: **source records**, **resolved loans** (distinct selected
identifiers), or a **bounded estimate of distinct exposures** when
identity is ambiguous.

**As implemented.** The stage's own predicates
(`pipeline.filters.loan_drop_reasons`) are applied to two columns typed
by the loader's rules (the status as text, the balance as a number, each
missing where the loader reads the cell as missing). Stage 2 has two
reasons, which the audit splits into the nine populations:

| Stage 2 reasons | Population | What the audit adds |
|---|---|---|
| none true | Active, retained | nothing |
| `balance_not_positive` only | Active status, zero / negative / missing balance | the balance is 0, below 0, or missing |
| none (the loader stops, the stage never runs) | Active status, balance unparseable | the loader's cast fails |
| `status_not_active` | Active status not recognised; Inactive; Inactive status not recognised; Eligibility unknown | the status against the code list, exactly and after trimming and upper-casing |

When the pipeline returned, the labels are compared record by record
with `Stage2Output` and called *observed*; a difference is shown as
"reconciliation incomplete" with both figures. When the pipeline raised,
they are *predicted from the Stage 2 predicate* and say so. Three
consequences of mirroring the stage: a balance of `NaN` is "Active,
retained"; a status of `ND5` is "Eligibility unknown"; with the status
column absent every record is "Eligibility unknown", and with the
balance column absent an active record is "balance missing".

Identity confidence: a duplicate is the same raw value on another
record; a collision is a different raw value equal after trimming, or a
value equal to a different record's value in the other identifier
column. Without a Stage 3 result the label says the selection is not
available. The findings themselves (A7, A12, A13) are family A's.

Lifecycle (3.3): a default status with a valid RREL9 is final-inactive
only when the date is not after an established cut-off; otherwise it is
"default, recoveries not determinable", and the evidence says why.

### 8.2 The reconciliation waterfall

Shown on the Summary sheet and the page, in records and in balance (per
currency, 8.5):

1. Records submitted.
2. Less: eligibility unknown; inactive, with "inactive status not
   recognised" (8.1) shown as a labelled part of the inactive subtotal.
3. Active status before the balance filter.
4. Less: zero, negative, missing balance; status not recognised (shown
   beside, since Stage 2 never saw it as active).
5. Active, retained (Stage 2 output).
6. Less, by exclusion point after Stage 2: no collateral record at all;
   collateral records only of excluded types (E1); property rows excluded
   for missing property type (E2) or no valuation (E3); selected
   identifier missing on the record; no match under the selected
   identifier convention; removed with a group by a later stage.
7. Loans in the MILAN output.

A loan can have several exclusion problems, and a stage can apply a
combined test: a loan with two collateral records, one of an excluded
type and one without a property type, is lost because no usable record
survives, not because of either alone. The waterfall therefore assigns
each lost loan to one line by a fixed **bookkeeping priority** (the order
of steps 4 and 6), stated on the report as a convention and not as the
established cause, so the lines add up to the loans lost; the Dropped loans
list shows that first cause and every other cause observed. Each line
also says whether it was observed from a stage result or predicted from
the stage's predicate because the stage did not run.

The **Dropped loans** list has one row per loan lost at step 4 or 6, with
the exclusion point, the reason and the collateral records seen. Unknown
eligibility and unknown amounts stay visible as their own lines; they are
not folded into "inactive".

Collateral counts are labelled as what they are: raw collateral records,
distinct collateral identifiers, property rows retained, properties
linked to a retained loan, graph groups.

The Execution Summary describes the post-join output population. G8
reconciles the audit's step 7 with it and step 5 with the Stage 2 result;
it does not require pre-join and post-join figures to be equal.

**As implemented.** Line 6 has one more part than the list above: **a
collateral record without a collateral type**, labelled "the Python
pipeline drops such a record today; the R reference keeps it" (the
difference recorded in `pipeline.filters.collateral_drop_reasons`). A
loan can be lost for it, so it needs a line. The bookkeeping priority of
line 6, in order:

| Order | Part | Test on a loan lost after Stage 2 |
|---|---|---|
| 1 | no collateral record at all | no collateral record links to the loan under either convention (raw match) |
| 2 | collateral records only of excluded types | every record linked under the selected convention is `collateral_type_excluded` |
| 3 | a collateral record without a collateral type | a linked record is `collateral_type_missing` |
| 4 | a property record without a property type | a linked record is `property_type_missing` |
| 5 | a property record without a valuation above 0 | a linked record is `no_positive_valuation` |
| 6 | the selected identifier is missing on the loan record | the loan is in `Stage3Decisions.loans_missing_calc_loan_id` |
| 7 | no match under the selected identifier convention | collateral records link to the loan only under the other convention |
| 8 | removed with a group by a later stage | no stage after Stage 3 removes a loan today; 0 |

Parts 6 and 7 cannot occur on a completed run until #45 is fixed: where
a collateral record refers to a retained loan under the column Stage 3
did not choose, Stage 3 raises. A loan Stage 3 dropped that none of the
tests explains is shown as "cause not established (reconciliation
incomplete)", never under a cause.

Observed and predicted. When the pipeline completed, lines 2 to 5 are
observed where the audit's retained loans are exactly Stage 2's; line 6
is observed from `Stage3Decisions`, which names every loan, with the
stage's own reason for every collateral record; line 7 is the rows of
the flattened pool. When the pipeline raised, the stages that completed
before the failure are read from the exception (`partial_result`,
`runner.run_pipeline`): a failure after Stage 2 leaves lines 2 to 5
observed, a failure after Stage 3 leaves line 6 observed. Otherwise
lines 2 to 5 are predicted from the stage's predicate; of line 6 the
first five parts are predicted, as the loans that would be lost under
either identifier convention, and parts 6 to 8 are "not available"; line
7 is "not available: the pipeline did not complete". An unparseable
balance is always predicted, because the loader stops at it.

Balance. Each line gives, per currency of RREL30 (its companion, else
RREL28; else "currency not stated"), the sum of the balance **as the
pipeline's loader reads it**, which is the number the pipeline itself
adds up; the report says so, and the exact text of every balance is in
the detail files. Records with no readable balance are counted apart.
Totals are never added across currencies.

The dropped-loans list has, per loan lost at line 4 or 6: the record and
its line in the file, the identifiers as written, the status, the
waterfall line, the assigned part, every other cause seen (including "a
collateral record of an excluded type" and "a collateral record matches
only after trimming whitespace"), observed or predicted, the balance
text and currency, and the record numbers of its collateral records and
of those Stage 2 kept.

### 8.3 Critical checks

The critical-field list (`DQ_CRITICAL_FIELDS`): RREL2, RREL3, RREL4,
RREL5, RREL6 / `pool_cutoff_date`, RREL23, RREL24, RREL29, RREL30,
RREL42, RREL43, RREL66, RREL67, RREL68, RREL69, RREL82; RREC2, RREC4,
RREC3, RREC5, RREC9, RREC13, RREC14, RREC15, RREC17, RREC18, RREC19.

The list now includes RREC3, which is a candidate in property-identifier
selection (one answer for the table in 6.3 and the constant).

On a critical field, the per-column rules that count are P1, P3a to P3e,
P4(d), P5 and P11, and only their violation results. P7(b) (capitals,
ASCII), P8, P9 and P12 never count.

**P6 on critical fields.** Until revision 8 P6 did not count at all, so
100 retained loans with a negative RREL29 and nothing else wrong would
be "eligible with warnings" with a compliance violation on every loan.
That consequence is now an explicit choice (assumption h in section
15). Proposed: P6 does not count as a whole; these sub-rules count on
these critical fields, and nothing else in P6 does:

| Sub-rule | Fields | Shares a defect key with |
|---|---|---|
| P6(a) amount below 0 | RREL29, RREL67, RREC13, RREC17 | E3 only where E3's own predicate holds on the record (both valuations missing or not above 0). RREC13 = -1 with RREC17 = 100 is a P6 violation and no exclusion: the record is used, and nothing is reported as excluded. |
| P6(b) date later than the cut-off | RREL23, RREL66, RREC15, RREC19 | C3 (RREL23), C8 (RREL66), E6 (valuation dates) |

A defect seen by P6 and by a cross-column check is one defect in rules
2 and 4. A10(a) is a pool-level fact and does not count. The
alternative is that P6 never counts, which keeps reporting compliance
and MILAN acceptance fully apart. P7(a), excessive length, is proposed
to count on critical fields (so "format" for RREL82 means presence and
length); that is a policy choice, question 9. For RREL82 nothing else
counts:
criticality does not turn every unusual trait of a critical field into a
contradiction. Cross-column checks count where section 7 says yes.

### 8.4 Counting without inflation

- Rule 4 counts **independent defects**, by `defect_key`, not findings.
  Rows are **filtered first, then unioned**. For the rejection rules the
  eligible rows are violations and policy triggers, from counting
  checks, on retained loans; for the compliance judgment, violations
  from rules with a regulatory source. Context, accepted alternatives,
  questions and limitations never enlarge an affected set, even when
  they share the key. A defect's affected set is the union of the loans
  of the eligible rows sharing the key, and that one union (loans and
  balance) feeds rules 2 and 4, whichever finding exposed the problem. The key
  assignments are a reviewed table in the catalogue.
  A missing property type is one defect whether it surfaces as P3f on
  RREC9 or as E2. P rules on one field that fire for the same cells
  (an invalid marker is P3d only, never also P5) share a key.
  A5 and P5 are the exception that keeps two keys for one cell: a value
  the loader cannot convert is a processing stop (A5, basis L) and, when
  it also breaks the prescribed format, a reporting breach (P5, basis
  E). The acceptance rules of a later release must not count that cell
  twice: where both keys hold the same cell, it enters an affected set
  once.
- Within a check, a loan counts once, with its balance once, however
  many of its collateral records are defective.
- A collateral finding is attributed to the loans that reference the
  record. It propagates to other loans of the group only where the
  pipeline's output for them actually depends on that record: the group
  valuation under aggregation, taken from the Stage 7 result. It is not
  propagated to every group member by default.
- Where identity is ambiguous the shares are a range. If the whole range
  is on one side of a threshold the rule is decided. If the range
  straddles the threshold (8% to 12% against 10%) the rule is
  **undetermined**; an undetermined rule gives the MILAN decision
  "Not determined: review needed", which withholds the MILAN file like a
  policy rejection and can be overridden like one. No endpoint is chosen
  silently. (Assumption e in section 15.)

**Composing the rules into one outcome**, in this order:

| Established facts | Policy outcome |
|---|---|
| any Stop | Blocked, whatever is uncertain elsewhere; never overridable |
| no Stop; an input to rules 2 to 4 is invalid (unexplained population difference, G8) or was never calculated (technical failure) | Unavailable; not overridable; nothing is released as eligible |
| no Stop; at least one rejection rule definitely met | Rejected by policy; remaining uncertainty is still shown |
| no Stop, no rule definitely met, and at least one rule undetermined | Not determined |
| every rule settled and none met | Eligible, or eligible with warnings |

Rule 4 composes the same way: five defects definitely above 1% is
rejection, and a further uncertain defect changes nothing; four definite
plus one uncertain is undetermined. Uncertainty can never turn a Stop or
a proven rejection into something weaker.

### 8.5 The four rules, restated

Denominators: loans and balance of "Active, retained". Comparisons are
strict and are test cases: exactly 10% does not meet "more than 10%",
exactly 1% does not meet "more than 1%", exactly 5% does not meet rule 3.
For a range: wholly above the threshold, met; upper bound at or below the
threshold, not met; lower bound at or below and upper bound above,
undetermined.

1. Any Stop.
2. A critical defect affecting more than 10% of loans or of balance.
3. More than 5% of balance lost at step 6 of the waterfall (balance
   only). Loans lost at step 4 are outside the rule's denominator by
   definition of the retained pool; that is a policy definition, not
   arithmetic. They are shown beside the rule in four groups: zero or
   negative balance; balance not measurable (missing or unparseable);
   **known positive balance excluded** (a status such as ` PERF`), with
   that balance totalled; and eligibility uncertain. A separate count-based measure for
   them (which populations, what threshold, how unknown statuses are
   treated) is a policy decision not yet taken (question 3).
4. Five or more independent critical defects each above 1% of loans.

Edge cases, stated on the report: a zero denominator (no retained active
loans) makes rules 2 to 4 "not evaluable" and the run is blocked by the
pipeline's own "no loans" or "no properties" fault.

**Currencies.** Stage 1 drops the currency companions and the pipeline
then adds, subtracts, divides and allocates amounts with no conversion.
Correctly reported cross-currency data is therefore valid reporting and
unsupported processing. Checking only the loan balances' currency is not
enough (all loans in EUR with a selected valuation in USD would pass), so
A17 covers every set of amounts the pipeline actually combines:

| Calculation | Amounts that must share one valid currency |
|---|---|
| pool totals, allocation weights, group balances | RREL30 of all retained loans |
| Ranking, External Prior Ranks, Pari Passu, Flexible Loan Amount | RREL30 with RREL31, RREL32, RREL33 |
| Total Income | RREL16 with RREL20 |
| Property Value, aggregation, calculated group LTVs (Additional data 10 to 12) | the valuations **actually selected** (from the Stage 7 result), each with its own currency evidence: a selected current valuation uses the RREC13 companion and RREC22; a selected original valuation uses the RREC17 companion only (RREC22 describes RREC13 and says nothing about RREC17). These must agree with each other and with the loans' RREL29 and RREL30. The currency of a valuation that was not selected never triggers A17; RREC22 against RREC13 is assessed on its own by C30. |

The reported MILAN Current LTV and Original LTV come from RREC12 and
RREC16 and are not recalculated, so they are outside this list. The exact
list is to be confirmed against the stages in the PR 1 research round.

Outcome, by the owner's decision (15.1, decision 3): where any of these
sets holds more than one currency or an invalid code, the audit
completes in full and the MILAN file **is produced with a warning**
(A17, Warn; it does not count towards rejection). No FX conversion is
added, so the affected figures are arithmetic on unlike units, and the
report says exactly that: which calculation combined which currencies,
on how many loans, properties and groups and for what balance per
currency, and which MILAN columns are affected (pool totals and
allocation weights, Ranking and prior ranks, Total Income, Property
Value and the calculated group LTVs). The warning is shown before the
download link, at the top of the report's Summary and on the summary
sheet appended to the MILAN workbook, so the file cannot be read
without it. A currency is **unstated
only when it remains unresolved after applying all permitted evidence
mappings** of 6.2 (the companion, then the documented dedicated field).
An empty or absent companion beside a valid RREL28 is a stated currency.
A genuinely unstated currency is a different case from a conflict: the
unit is unknown, the compliance finding is in 6.2 (case 4), and whether
that also blocks MILAN or is processed under a stated single-currency
assumption is decision 12 in 15.1; both existing fixtures would be
affected by a block. The report's own totals are shown per currency and never added
across currencies or across stated and unstated units.

---

## 9. Pipeline observation and output lineage

### 9.1 Use the pipeline's own decisions

The first version forbade any change to a stage module and would have
re-derived the filter, the identifier choice and the valuation choice
inside the quality package. Two implementations drift. Revision 2 asks
for a narrow, behaviour-preserving interface instead (its own PR, parity
gate applied):

- Stage 2 exposes its two predicates (loan retained; collateral row
  retained, with the reason a row fails) as reusable expressions, which
  the audit applies to its own frames to label populations. When the
  stage ran, the audit's labels are reconciled with the stage's output
  (G8).
- Stages 3, 6 and 7 return what they decided as data: selected
  identifier column, coverage and gate figures; aggregation convention,
  evidence and fallback. Valuation provenance is returned at the
  granularity at which it is decided, not one flag per group: per
  collateral record, which valuation (current or original) was preferred,
  whether the flip applied and what was selected; per property, the
  record(s) reduced into it; per group, the representative property and
  where its method, date and metadata came from; and the aggregation and
  allocation to loans. One group can use a current valuation for one
  property and an original one for another; lineage (9.3) and defect
  propagation (8.4) need every contributing record identified.
- The known data-fault sites raise a `PipelineDataError` (a subclass of
  `ValueError`, same message, so nothing that catches `ValueError`
  changes) carrying the stage, the reason code and the identifiers
  involved.

Diagnostics today go two ways: `warnings.warn` and `structlog` events
(the aggregation fallback and the encoding repair use the latter), and
their messages carry samples, not complete lists. So the structured
checks and the stage results above are the primary evidence; captured
warning and log text is attached to the matching finding as a supplement
and never used to build a row list.

### 9.2 When a stage fails

| What happened | Outcome |
|---|---|
| `PipelineDataError` at a known site (non-numeric value, no loans or no properties after filtering, identifier selection or coverage failure, duplicate identifiers) | MILAN decision "Blocked"; the fault becomes a Stop finding linked to the audit findings that explain it; checks that need a later stage are "not evaluated: stage N did not complete"; all file-level checks have already run |
| A Polars exception at a known site with a known meaning, in the loader or later (for example a column a stage expression needs, such as RREC17 or RREL4, is absent) | as above. The catalogue of expected data faults is a list of **sites and meanings**, compiled in the PR 1 research round by reading every stage; an exception is never classified by its Python type alone. |
| Output too large (`OutputTooLargeError`) | its own outcome: the assessment and policy result stand, the MILAN workbook is "not written: output capacity", reports remain |
| Any other exception, including a plain `ValueError` from elsewhere | **technical failure**: the job fails as today, the report produced so far is kept and marked "assessment interrupted by a technical fault"; it is not presented as a data rejection |

### 9.3 Lineage (P10)

A static field-to-column table cannot say why an output cell is "ND".
Each ESMA field gets a lineage class, and each statement is marked
**observed** (read from the produced MILAN pool) or **predicted** (the
pipeline did not get that far):

| Class | Meaning | Examples |
|---|---|---|
| Direct | one field to one output, possibly through a code map | RREL27 to Loan Purpose |
| Conditional | one of several sources, or a combination | Total Income (RREL16, RREL20); Borrower Type (RREL13, RREL17); Base Index (RREL44, RREL45); Restructured Loan (RREL65, RREL69) |
| Selected | chosen by valuation selection per collateral record, reduced per property, represented and aggregated per group (9.1), with fallback | Property Value, Property Valuation Type and Date |
| Group-derived | depends on other records of the group | Ranking, External Prior Ranks CB, Pari Passu, Property Value under aggregation |
| Gate | decides whether the record exists in the output | RREL69, RREL30, RREC5, RREC9, RREC13/17, the identifiers |
| Default | a missing input yields a constant or a fallback, not "ND" | Pari Passu "0", Flexible Loan Amount "0", Social Programme "6" (and "5" for unrecognised text), MIG Provider "No Guarantor" |
| Not used | no effect downstream | most RREL44 to RREL64 fields |

A field can sit in several classes and have a different consequence in
each output it feeds (RREC23: "No Guarantor" in one column, "ND" in
another). Indirect effects are part of the lineage: balances drive the
allocation and the calculated LTVs; occupancy takes part in choosing the
representative property; the cut-off drives seasoning and Months Current;
identifier fields drive matching and grouping as well as their own output
columns; RREL4 and RREC3 feed identifier generation.

Output columns that are constant by design (Property 2 to 5 blocks,
Servicer Identifier, the literal "ND" columns) are listed once as
"constant in this pipeline" and excluded from the "more than 95% ND"
warning, which applies only to columns with a Direct, Conditional or
Selected source.

---

## 10. Part 3: how the report is shown, and the result contract

### 10.1 The result contract

Today a workbook job returns one output path, the worker asserts it
exists, the result endpoint serves the binary workbook, and the analysis
builders assume every stage completed. A report with no MILAN file fits
none of that, so the contract changes in one PR together with its
consumers (11.3). A run's result separates:

| Field | Values |
|---|---|
| `execution` | completed / technical_failure |
| `assessment_completeness` | complete / partial / file_unreadable |
| `compliance` | two fields: `violations` (none / present, with counts) and `coverage` (full / partial, with counts). Both can be non-trivial at once. Coverage here is of the compliance checks only; missing pipeline-dependent analysis is reported under `assessment_completeness` and does not lower it. A report-rendering failure after a completed assessment is an artifact state, not incomplete coverage. |
| `assessment` | ran / interrupted (technical) |
| `policy_outcome` | eligible / eligible_with_warnings / blocked / rejected / undetermined / **unavailable** (a technical failure happened before the outcome could be calculated; with the reason). Never defaulted to eligible, never turned into a data rejection, and distinct from `undetermined`, which is an uncertainty range and is the only overridable uncertain state. An outcome already established before a technical failure is kept, unless the failure invalidates the evidence it rests on (G8), in which case it becomes unavailable. Plus `release_permitted` (true for eligible and for eligible_with_warnings; true for rejected or undetermined only with a valid applied override; false for blocked and unavailable) and `override_applied`. Eligible with warnings therefore releases the workbook, answers 200 on the synchronous route and exits 0, with the warnings shown before the download link. This is permission, not production. |
| `requested_outputs` | which outputs the run asked for: MILAN workbook, analysis, reports only (a dry run and the analysis job request no workbook; today's analysis worker runs with `dry_run=True`) |
| `override` | requested (bool), applied (bool) |
| `artifacts` | a manifest, one entry per output: name, kind, state (**available**, **not requested**, **withheld by policy**, **failed** with reason, **incomplete**), size, media type. "MILAN produced" is read from this manifest, never from the policy outcome. Covers a workbook-write failure after a successful computation, and JSON and CSV succeeding while the report workbook fails. An incomplete file is never offered as a download. |
| `analysis` | present only when the pipeline completed |
| `provenance` | section 10.4 |

Behaviour per entry point:

| Entry point | When release is not permitted by policy (blocked; rejected or not determined without a valid applied override). A technically unavailable outcome is the last row. |
|---|---|
| Background workbook job | job state `done`; when release is not permitted the manifest shows the MILAN workbook as withheld by policy and the binary result endpoint answers 409 with the summary and points to the manifest; with a valid applied override the workbook is available and the underlying outcome is unchanged |
| Background analysis job | `done`; `analysis` absent; the page shows the data quality section alone. The analysis builders are not called when the pipeline did not complete. |
| Synchronous endpoint | one contract, below: no artifact links |
| CLI | writes the report files. Exit codes describe the run, not whether a workbook was requested, and follow the release permission: 0 release permitted (eligible, eligible with warnings, or rejected or not determined with a valid applied override) and every requested output written, including reports-only and dry runs; 2 rejected by policy and still withheld (no valid override); 4 not determined and still withheld; 3 blocked by a Stop (the override flag changes nothing); 5 release permitted but a requested artifact could not be written (output capacity or an export failure), with the manifest printed; 1 technical failure, including an unavailable policy outcome. The report always states the underlying outcome, overridden or not. |
| CLI dry run | runs the audit and the pipeline, writes the report files only |
| Any entry point, policy outcome unavailable (a technical failure before the outcome, or G8) | A technical failure, not a policy result. Background job: job state `failed` with the technical reason; `execution` is technical_failure; `policy_outcome` is unavailable; the manifest shows the MILAN workbook as **failed** with that reason, never "withheld by policy"; every finalised report artifact stays listed and retrievable; the binary result endpoint answers as for any failed job, with the technical reason and a pointer to the manifest, and not with the 409 policy response; no "Process anyway" action is offered. Synchronous: 5xx with the reason. CLI: exit 1. |
| CLI, several conditions at once | one exit code, by precedence: 1 technical failure or unavailable outcome; 3 Stop; 2 rejected and withheld; 4 not determined and withheld; 5 artifact failure; 0. So a rejected run without an override whose report workbook also failed exits 2; a Stop with an artifact failure exits 3; an overridden rejection exits 0 when everything was written and 5 when an artifact failed. The manifest and the printed summary keep every fact. |

**Synchronous requests: one contract.** The synchronous handler works in
a temporary directory removed when the response is sent, so it has no
retrievable artifacts. Proposed, as a deliberate product choice
(question 10):

| Synchronous call | Response |
|---|---|
| workbook, release permitted (eligible, eligible with warnings, or rejected or not determined with a valid applied override) | 200, the binary MILAN workbook as today; it carries the appended DQ summary sheets (10.3), which state the underlying outcome and the override. No JSON body. |
| workbook, release not permitted (blocked; rejected or not determined without an override) | 409, JSON: the quality summary (judgments, triggered rules, counts, waterfall) and the sentence that the complete affected-row report is produced by the background job route or the CLI. No artifact links. |
| dry run or analysis | 200, the existing JSON with the quality summary added; `analysis` absent when the pipeline did not complete |
| technical failure, or policy outcome unavailable (including G8) | 5xx as today, with the reason; no workbook |

So "a complete report on every run" holds for background jobs and the
CLI. A synchronous caller who is blocked or rejected gets the summary and
must submit a job to get the row-level report. If that narrowing is not
acceptable, the alternative is to persist synchronous assessments under
an assessment id with the same retention, which is more work.

**Assessment identifier.** Every run has an assessment id that does not
depend on an API job (the CLI has none): derived from the input hashes,
the rule-set version and the start time. File names and provenance use
it.

**Failed and interrupted jobs.** A job that fails technically, or is
interrupted and recovered by the worker, exposes whatever report
artifacts were finalised, through the same manifest, and the page shows
them with the failure. Files are written to a temporary name and renamed
when closed, so a half-written file never appears as available.

All routes are under `/api/jobs`, as today. Artifacts are retrieved by
name under the job (`/api/jobs/{id}/artifacts/{name}`), expire with the
job's 12-hour retention, and are finalised (written and closed) before
the job is marked `done`. Report file names use the cut-off date when
there is exactly one, else `unknown-cutoff`, plus the assessment id (the
same for API jobs and the CLI).

**Data for the page.** A download link cannot feed tables, so the API
serves bounded views:

- `GET /api/jobs/{id}/quality` : judgments, completeness, waterfall,
  rule outcomes, counts, provenance.
- `.../quality/fields` and `.../quality/columns`: the two inventories.
- `.../quality/findings?basis=&severity=&family=&field=&counts=` :
  findings without rows, with a stable `finding_id`, total count,
  defined ordering (severity, counts, records affected, id).
- `.../quality/findings/{finding_id}/rows?offset=&limit=` : pages of at
  most 200 records, with the total.
- `.../quality/dropped?offset=&limit=` : the dropped-loans list.

Bounds: default page 50, maximum 200 records; rows ordered by file, then
record number, then physical column; a cell longer than 2,000 characters
is cut in the API with its full length and a pointer to the detail file;
a response is capped at about 2 MB and returns fewer rows with a
continuation offset if needed; after the job expires the endpoints answer
404 as every job route does today (cleanup removes the whole job
directory, so expired, deleted and never-existing jobs cannot be told
apart without keeping extra evidence; not proposed). Columns are identified by file and physical position, so two columns
with the same header stay distinct.

A finding group is **homogeneous by result kind**: `finding_id` is
`check_id`, sub-rule, result kind, file and field joined, so C9a's
violations and its context rows are two groups with their own counts,
and the kind is also kept on every evidence row through storage, API,
Excel and CSV. Example: `P3d:violation:loans:RREL16`. `check_id` alone is
not unique. There is no jobs
list in the application today, so the verdict badge on a list is dropped
from this plan.

### 10.2 Web page

A **Data quality** section, shown whether or not the pipeline completed:

1. **Four tiles**: Reporting compliance, MILAN decision, Plausibility,
   Assessment completeness, each with its count and colour. Under the
   MILAN decision, the rules that triggered in plain words, each with its
   figure and threshold; for a Stop, the message, the fix and "cannot be
   overridden".
2. **Waterfall** (8.2): submitted to MILAN output, with the lost loans
   and balance at each step.
3. **Downloads**: report workbook, detail CSV; MILAN workbook when
   produced. When rejected by policy or not determined: **Process anyway**, with a
   sentence matching the outcome: "Produces the MILAN file despite the
   rejection. The rejection and every finding stay on the report." or
   "Produces the MILAN file although the outcome could not be
   determined. It stays 'not determined' on the report." The button is
   never offered for a Stop or when the policy outcome is unavailable
   after a technical failure. It resubmits the
   files the browser still holds; if they are gone the button is
   replaced by "Upload the same files again and tick 'process despite
   rejection'".
4. **Tabs**: *Fields* (the 107), *Columns* (as supplied), *Findings*
   (filters: basis, severity, family, field, counts towards the decision;
   rows on demand), *Dropped loans*, *Checks* (every check with its
   outcome, so "not evaluated" is visible), *Pool figures*.

Warn counts sit above the download link, as decided.

### 10.3 Files

The detail (one row per finding and record) can run to millions of rows.
It is never held as Python objects. Checks produce Polars frames of
(audit key, physical column position, field, finding id), so two
supplied columns that resolve to one field never share evidence; a
cross-column finding lists each contributing cell by record and physical
column. The detail is assembled lazily and
streamed to disk in the job directory, then read in pages by the API and
in one pass by the exporters. Write-only Excel bounds only the worksheet
writer; the raw text frames and the typed pipeline frames are still in
memory, so peak memory is roughly today's plus one all-text copy of each
file. That is to be measured on the largest sample pool during
implementation, not assumed.

| File | Content | Fidelity |
|---|---|---|
| `... data quality detail.csv` | every finding row: finding id, result kind, basis, severity, file, record number, physical column position, original header, field, value state, raw value, parsed ND4 date, expected, raw identifiers, population, counts | **Preserves the defined cell-level evidence, reversibly**; it is not a byte-for-byte copy of the source file (quoted and unquoted empties are not distinguished). Encoding contract: UTF-8; a backslash in the source is written as two backslashes, a control character as backslash-x and two hex digits, so the literal four-character text backslash-x-1-F (which becomes `\\x1F`) and the character 0x1F (which becomes `\x1F`) stay different and the original can be recovered exactly; a `raw_encoding` column says plain or escaped; the original bytes of repaired cells are in a separate hexadecimal column; the first line carries the schema and escape version. |
| `... data quality report.xlsx` | `DQ Summary` (judgments, rules, waterfall, provenance, versions), `DQ Fields`, `DQ Columns`, `DQ Checks` (outcome of every check), `DQ Findings`, `DQ Rows`, `DQ Dropped loans` | A rendering. Values that Excel cannot hold are sanitised and marked: XML-illegal characters shown escaped; text beyond 32,767 characters cut with "… (n characters; full value in the detail CSV)"; every raw value and every identifier written as a text cell (leading zeros and long digit strings survive), counts as numbers, nothing as a formula, so a value starting with `=` stays text |
| Sheets appended to the MILAN workbook | `DQ Summary`, `DQ Fields`, `DQ Columns`, `DQ Checks`, `DQ Findings` only | Abbreviated by design: no row sheets. The Summary states the total row count and the name of the complete file. The two outputs are not described as identical. |
| `... dropped loans.csv` | the complete dropped-loans ledger: one row per lost loan with the assigned waterfall line, **every** observed cause, balance and currency, and the record numbers of its related collateral records | Complete, same encoding contract. This, with the detail CSV, is what "the CSV remains complete" means: the detail CSV holds every finding cell, this file every lost loan and its relationships. |
| `<workbook name without extension> data quality summary.json` | the `/quality` payload. Written by the command line beside the workbook, on a dry run and when the pipeline raises; `unknown-cutoff` takes the cut-off's place in the name when none is established. It holds the provenance, the judgments, every rule's outcome, the finding groups (counts, no rows), the two inventories and the population counts. Until the result contract of 10.1 is implemented, `execution` is `completed` or `pipeline_failed` (a fact about the run; a typed data error is given with its stage, reason code and message, any other error with its type only and the words "an error that carries no reason code"), every finding group has a sentence built from one clause per rule and result kind (`catalogue.SENTENCES`), the later parts say "not implemented in this release", and no policy outcome is stated. | |

Overflow: `DQ Rows` and `DQ Dropped loans` continue on numbered sheets
past 1,048,575 rows; above `DQ_XLSX_MAX_DETAIL_ROWS` in total the
workbook holds the first rows per finding, says so with included and
total counts, and the CSV remains complete. If the report workbook
cannot be written at all (size or a serialisation fault), the CSV and
JSON still exist, the job reports the workbook as unavailable with the
reason, and the outcome is not a silent success. Report sheets are plain
values like the existing ten (the current writer emits values only, no
bold, freeze, filter or widths); adding a header style to the report
sheets alone (bold, frozen row, filter) is adopted, per the second review.

**As implemented (command line).** Four files are written beside the
MILAN workbook, each named with the workbook's name without its
extension as the stem:

```
<stem> data quality detail.csv
<stem> dropped loans.csv
<stem> data quality report.xlsx
<stem> data quality summary.json
```

The detail store. The assessment writes every result row to Parquet
parts in a working directory unique to the run, created inside the
output folder (in the system temporary directory if that cannot be
written) and marked with a file that holds the process id; a marked
directory left by a process that is no longer running is removed at the
start of a later run, and a directory without the marker is never
touched. With the rows goes an index of counts (rows and records per
file, field, rule, kind, severity and reason), from which the summary is
built without reading the detail. The directory is removed at the end of
the run, also after a failure and after Ctrl-C.

Two processes of their own. The files are assessed before the pipeline
runs, in a fresh interpreter that writes the store and exits; the report
is completed and its files are written after the pipeline, in another
fresh interpreter that is handed the assessment and the facts of the run
(the stages' decisions and counts, no frame of the pool). The
conversion's process therefore holds neither the audit's memory nor the
exporters'. Each child prints nothing and logs nothing. A child that
dies is a fault of the report: one log event with the step and the exit
status, the summary says what is missing, and the pipeline's run, output
and exit status are what they are without the report.

The two CSV files start with a schema line, then the header:

```
# esma-milan data quality detail; schema 1; escape 1
finding_id,rule_id,result_kind,basis,severity,file,record,physical_line,column_position,header,field,value_state,raw_value,raw_encoding,original_bytes_hex,nd4_date,expected,reason,note,loan_identifier_original,loan_identifier_new,obligor_identifier_new,collateral_identifier_new,population,counts

# esma-milan dropped loans; schema 1; escape 1
record,physical_line,loan_identifier_original,loan_identifier_new,obligor_identifier_new,status,waterfall_step,assigned_line,other_causes,observed_or_predicted,balance,currency,collateral_records,collateral_records_kept_by_stage2
```

The detail file is ordered by file (loans, collateral, pool), record
(rows about a whole column or file first), column position and rule. The
identifiers are those of the row's own record: RREL2, RREL3 and RREL5
for a loan record; RREC2 (as the new loan identifier) and RREC4 for a
collateral record. `population` is the loan's label; for a collateral
record, the labels of the loans it links to. `counts` is `yes` when the
row is a violation of a counting rule on a critical field; it is the
catalogue's flag and nothing is enforced in this release. In
`other_causes` and the two collateral-record columns, entries are
separated by `;`.

Escape 1, applied to the columns that can carry text of the user's files
(`header`, `raw_value`, `expected`, `note`, the identifiers; in the
dropped-loans file the identifiers, the status, the balance and the
currency): a backslash is written as two backslashes; every character
below U+0020, and U+007F, as backslash, `x` and two hexadecimal digits.
`raw_encoding` is `escaped` when a column of the row was changed by this
and `plain` otherwise. To read a value back, left to right: `\xNN` is
the character NN and two backslashes are one.

| In the file | In the CSV | `raw_encoding` |
|---|---|---|
| `AB` with the character 0x1F between | `A\x1FB` | escaped |
| the six characters `A\x1FB` | `A\\x1FB` | escaped |
| `C:\dir` | `C:\\dir` | escaped |
| two lines `x`, `y` in one quoted cell | `x\x0Ay` | escaped |
| `=1+1` | `=1+1` | plain |
| `0012` | `0012` | plain |

A cell whose bytes were not valid UTF-8 shows the repaired text in
`raw_value` and its original bytes in `original_bytes_hex`.

The report workbook is written with openpyxl in write-only mode by
`quality/report_workbook.py`; `io_layer/` is not involved. Sheets, in
order: `DQ Summary` (section, item, value, balance, basis), `DQ Fields`,
`DQ Columns`, `DQ Checks`, `DQ Findings`, `DQ Rows` (the columns of the
detail CSV) and `DQ Dropped loans`. Bold header, frozen first row and a
filter on each. Every value of the user's files is a text cell and no
cell is a formula; a control character is shown escaped as in the CSV;
text beyond 32,767 characters is cut with its length and the name of the
detail CSV. `DQ Rows` continues on `DQ Rows 2`, and so on, past a
sheet's row limit. Above `DQ_XLSX_MAX_DETAIL_ROWS` finding rows (500,000:
measured, a workbook of that many rows adds under two minutes to a run
on a 500,000-loan file) `DQ Rows` holds the first rows of each finding
group in proportion to the group's size, every group keeping at least
one row, and `DQ Summary` gives the rows included, the total and the
name of the detail CSV. Only this rendering is ever cut: nothing is
sampled or dropped in the store, the detail CSV or the summary.

| Run | Summary | Two CSVs | Report workbook | Working directory |
|---|---|---|---|---|
| normal, dry run, pipeline raised | written | written | written | removed |
| interrupted (Ctrl-C, `SystemExit`) | none | none | none | removed |
| a file cannot be written | written, with that file's state `failed` and the reason | the others are written | the others are written | removed |
| the report-writing process dies | written by the main process, saying the other three are missing and why | absent | absent | removed |
| the assessment did not complete | written, saying so | none | none | removed |
| no working directory can be created | none; one log event with the step and the exception's type | | | |

Each file is written to a temporary name and renamed. The summary is
written last, and its `detail_files` block gives each of the other
three: name, state, size, rows, and for the workbook the sheets and the
rows included. The summary (schema 2) also gained `waterfall`,
`dropped_loans` (counts by line) and `links`, and the outcomes of family
A, G8 and A17 in `rules`.

Not built yet: the sheets appended to the MILAN workbook (the row above
in the table); the job directory and the paged reads of the API. A later
PR that keeps the store in a job directory, where another party could
write, must hand results between processes in a plain data format and
not as a pickle (`docs/decisions.md`).

### 10.4 Provenance and overrides

Every report records: SHA-256, name and size of both input files; the
effective options; pipeline version (commit), rule-set version, taxonomy
hash, ESMA package version, reference-list versions; every effective
threshold; assessment time and the as-of date used for "today"; the
judgments and triggered rules.

An override is a new run with the flag. All checks run again, a Stop
still blocks, and the report carries `override requested` and `override
applied` plus the parent assessment's id and judgments when the file
hashes and options are identical. If the hashes or options differ it is a
new assessment and is shown as such, not as the previously reviewed pool.
The original rejection stays on the report.

Details: the client supplies the parent assessment id with the override;
the server accepts it as a parent only if that assessment still exists
and its file hashes and options match, otherwise the override run stands
alone and says "no verified parent" (an expired parent is not an error).
The options compared are those that change the output (deal name,
aggregation setting, minimum coverage), never the override flag itself.
"Same files" and "same assessment conditions" are reported separately:
the same hashes assessed under a different rule-set, reference-list,
threshold set or as-of date is a re-assessment of the same files, and
the differences are listed. The build's commit and version are written
into the package at build time (the Docker image has no Git checkout).

### 10.5 The parity guarantee, stated precisely

The parity machinery compares workbook contents sheet by sheet with its
established canonicalisation and tolerances; it does not prove byte
identity of the `.xlsx`, and appending sheets changes the package. The
guarantee is therefore: **the ten canonical sheets, in their order, keep
passing the existing comparison unchanged.** The new diff option
requires all ten canonical sheets to be present, in the canonical order,
as the first ten; compares them in full; and allows extra sheets only
after them. It does not compare "common sheet names". Two caps exist in the
diff and only one is harmless. `max_diffs_shown` limits printing.
`max_cell_diffs_per_sheet` (default 50) stops the comparison of a sheet
once that many differences are recorded, and the real-fixture test then
removes tolerated last-digit differences from the recorded list: if the
first 50 are all tolerated, a later material difference is never
assessed and the test passes. A cap of zero or less returns before the
first mismatch is recorded. Since parity is the assurance that the stage
interface of 9.1 changes nothing, PR 1 must first make pass or fail
independent of any cap (compare every cell; cap only the examples kept),
keep both existing tolerance layers with exactly their present scopes,
so nothing more and nothing less counts as parity than today, and add a
test with a material difference placed after more than
50 tolerated ones. The two layers, checked in the
code: (1) the general numeric-cell tolerance of the diff library
(`NUMERIC_ABS_TOL` 1e-9, `NUMERIC_REL_TOL` 1e-12 in
`src/esma_milan/parity/diff.py`), which applies to every numeric cell of
every sheet; (2) the real-fixture layer in
`tests/parity/test_parity_real.py`, which accepts a difference between
two strings that parse as floats within relative 1e-13 and absolute 0,
only in the six listed MILAN columns. All other string cells compare
exactly, subject to the existing canonicalisation. The fix moves layer 2
inside the comparison without widening it and leaves layer 1 as it is;
the general numeric tolerance is not replaced by exact comparison. This
is a defect in
today's gate regardless of this work and deserves its own issue. The report sheets are excluded from the R comparison, so they get
content tests of their own (11.2).

**Parity stays testable when policy would reject a fixture.** The parity
fixtures are not compliant examples, and the parity tests call
`run_pipeline` and require a workbook. So the conversion and the release
decision are kept apart in code: `run_pipeline` always performs the
conversion and returns its result and the quality report; the policy
outcome is applied by the release layer (API handlers, worker, CLI),
which decides whether the workbook is written out or withheld. Parity
tests exercise the conversion; separate tests assert that the release
layer withholds or releases correctly for each outcome. Neither the
rules nor the reference fixtures are weakened. A pipeline data fault
(no workbook can exist) is unaffected by this.

---

## 11. Implementation sketch, tests and PR split

### 11.1 Package

`src/esma_milan/quality/`: `catalogue.py` (reviewed field rules with
sources, code lists, conditional rules, lineage classes), `refdata/`
(ISO 3166-1, ISO 4217, NUTS 3 by year, as small data files), `audit_read.py`
(5.1), `states.py` (value states and audit parsing), `columns.py` (P
rules, inventories), `rules/` (families, one function per check, each
declaring basis, applies_to, needs, defect_key), `populations.py`
(section 8), `lineage.py` (9.3), `decision.py` (the four rules,
override), `store.py` (on-disk detail, paging), `export.py` (CSV, xlsx,
JSON), `result.py` (the contract of 10.1). Thresholds, the critical
lists and the placeholder lists are `config.py` constants.

As built so far: `catalogue.py`, `refdata/`, `audit_read.py`,
`states.py`, `column_rules/` (`nd.py`, `formats.py`, `domain.py`,
`currency.py`, `plausibility.py`, with the engine in its `__init__`; it
takes the place of the single `columns.py` above), `inventories.py`,
`populations.py`, `result.py` (result rows and coverage counts),
`assess.py` (`assess_files`, `complete_report` and the report object),
`export.py` (the summary), `store.py` (the on-disk detail and its index
of counts), `child.py` (the two child processes), `links.py` (5.2),
`family_a.py` (7.1), `waterfall.py` (8.2, with G8 and A17),
`detail_csv.py`, `report_workbook.py` and `report_files.py` (10.3), and
`column_rules/fast.py` (the fast path: values proven plain by a
conservative test are grouped by their other cells only and the rules
run once on a representative; a test compares it with the slow path on
the fixtures, at every boundary and on seeded random columns).

### 11.2 Tests

The two existing fixtures are parity fixtures, not compliant reporting
examples. The synthetic one uses cleaned headers and `NA` markers and
carries only 60 loan and 13 collateral columns (RREL1, RREL6, RREL28,
RREL83, RREL84 and RREC1 are among the absent fields); the real one has upper-case
`TRUE` in RREL10 and RREL14 and lacks RREC22. The test for each is its
**reviewed, exact list of findings**, checked in as an expected file and
reviewed by the owner once; rules are not weakened to make them quiet.

New test data lives outside `tests/fixtures/` (which stays untouched),
built by test code: one small compliant pair of files that must produce
no Stop and no Warn of any basis (Info context such as lineage and
profiles is expected), and targeted cases:

- independent defects in both files reported despite an early numeric
  error;
- valid and invalid dated ND4; ND4 date not after cut-off;
- conditional ND5 on property and non-property collateral; OTHR and MIXD
  left unevaluated;
- negative margin and negative allocated loss accepted;
- exposure-specific original valuations on a shared property accepted;
- duplicate, missing, padded and colliding identifiers; trimmed collision;
- no property rows surviving the filter; no loans surviving;
- many defective collateral records on one loan counted once;
- unknown eligibility; unparseable and negative balances; ` PERF`;
- two currencies;
- rejected, overridden and blocked outcomes through the API job, the
  synchronous endpoint and the CLI, including exit codes;
- a large generated export for memory and time; control characters, a
  40,000-character cell and a value starting with `=` through the
  exporters;
- every check reporting an explicit outcome, including not evaluated;
- decimal precision cases (`1.234560000`, `1.234567`, `+01.5`, `1e3`);
- report-sheet content tests, and the ten-sheet prefix rule;
- the false-positive cases of the reviews. Each asserts three things
  separately: the prohibited conclusion is absent (no violation, no
  contribution to rejection); the expected processing or context result
  is present; nothing else is required to be silent. `NA` as a country
  code: no compliance violation, and the L finding that the loader loses
  the value. A re-performing loan keeping default amount and date:
  context only. A lien position on a guarantee: nothing. A valuation
  amount with ND1 for its method: an availability context row, no
  violation. A verified zero secondary income: nothing. Pro-rated DTI on
  two loan parts: no C47 row. An original LTV whose prior encumbrance was
  repaid: at most an Info note. Equal, correctly pro-rated amounts on
  shared collateral: facts only, no pro-rating conclusion;
- a valid identifier beginning `ND4` (`ND4000123`): no malformed-ND4
  finding and no rejection contribution;
- ` 12 `, `.5`, `1.` and ` true `: P4 and P5 agree (no P5 violation), raw
  cell kept; a non-breaking space is not collapsed;
- RREL65 with two valid dates (passes); with one valid date and one
  invalid fragment (fragment violation kept, pipeline date noted);
- RDMD with a positive balance: L1 question visible, the record stays
  final-inactive (presumed);
- DFLT with a valid default date and RREL71 = ND5: C43(b) violation; the
  same on a final-inactive record: none;
- original EUR valuation selected while the current valuation and RREC22
  are USD: no A17 warning;
- context and violation rows under one check and defect key: result kind
  survives export and only the violations enter the numerator;
- a Stop plus an undetermined rule (blocked, no override); a proven
  rejection plus an undetermined rule (rejected); four definite defects
  plus an uncertain fifth (not determined); five plus an uncertain sixth
  (rejected);
- an applied override through the synchronous endpoint (200 workbook,
  underlying outcome preserved);
- a technical failure before the outcome is calculated (`unavailable`,
  no override offered);
- `rdmd` and ` RDMD `: exactly one processing population each;
- lifecycle: a final-inactive property record with RREC5 = RBLD, RREC9 =
  ND5, RREC7 and RREC16 = ND5, RREC12 = 0 (no finding); the same pattern
  on a reporting-active loan (question plus the Stage 2 exclusion); a
  retained old valuation method on an inactive exposure sharing a
  property with an active one (no violation);
- currency: EUR loans with a USD selected valuation (A17 warning, file
  produced, affected columns named, warning present on the appended
  summary sheet); primary and secondary income in different currencies
  (A17 warning); companions absent
  and no dedicated field either (unstated unit, treated per decision 12);
- two physical columns resolving to one field, with different raw
  values, through finding generation, storage, export and API;
- LTV denominators of zero and below zero;
- positive arrears days with an unavailable last-arrears date (the L
  note says Months Current is "0");
- `ND5` and `ND4-2027-01-01` in a free-text field (ND states, not
  content); `1_000`, full-width digits and ` 12 ` in a numeric field;
- findings and partial coverage together surviving API serialisation;
- thresholds at exactly 10%, 1% and 5%; a range touching a threshold;
- an undetermined outcome through the CLI (exit 4, 0 with override) and
  the override button;
- a partial export failure (CSV and JSON written, workbook fails);
- publication and recovery: interruption after an artifact is renamed
  but before the manifest is updated; after the manifest is updated but
  before the job state is published; and during workbook export with CSV
  and JSON already final. After recovery a completed file is
  discoverable and an incomplete one is never advertised;
- Boolean inference: an all-`true`/`false` column against a mixed one;
- a fallback valuation that is itself at most 10;
- a check with findings and non-evaluable records together;
- an uncertainty range straddling a threshold;
- round-trip of the detail CSV encoding (0x1F against the text `\x1F`);
- a material parity difference after more than 50 tolerated ones;
- a numeric-cell drift within 1e-12 relative in a column outside the six
  (still passes, layer 1), and a string drift in a column outside the
  six (fails).

Added with revision 6. In the lifecycle cases "no finding" means no
inappropriate compliance or lifecycle finding; processing observations
(an exclusion, lineage) may still be present.

- RDMD with a valid RREL9 not after the cut-off and RREL30 = 0: no L1
  question, no C27 finding; RDMD with RREL9 = ND5: C27 violation;
- a final-inactive property record with RREC13 = 0 and a positive
  RREC17: no E8 warning, one context row; E9 not applicable;
- DFLT with RREL9 = ND5, a valid RREL72 and RREL71 = ND5 (default,
  recoveries not determinable): C43(b) violation;
- 100 retained loans, 11 with one excluded extra property record and a
  surviving one: 11 E2 warnings, none counted, rule 3 shows no loss; 11
  loans whose only property record is excluded: 11 counted once each in
  rules 2 and 4 and their balance in rule 3;
- a rejected run with a valid override through the CLI: exit 0; the same
  with a failed artifact: exit 5; without the override: exit 2;
- eligible with warnings: released on the job route, 200 on the
  synchronous route, exit 0, warnings shown;
- G8 fault injection: audit retained pool 100, pipeline 90, 10 affected
  loans (10% against 11.1%): outcome `unavailable`, no release, no
  override, finalised report files still served, per-column results
  still shown; through a background job also: job state `failed`,
  `execution` technical_failure, manifest reason technical and not
  "withheld by policy", result endpoint answering as a failed job;
- `ND4X0000000000000085` in an LEI field: a present, valid LEI; no
  malformed-ND4 finding;
- RREL65 with a valid date and an `ND5` element: two recognised
  elements, a question on the mixture, no invalid fragment; RREL65 with
  `ND4-2027-01-01` after the cut-off: a valid ND4 state, no "later than
  cut-off" violation;
- ` PERF` and ` GB `: one P5 violation each and nothing from P4(a);
  ` 12.5 ` and ` true `: accepted alternative, no P5 violation.

Added with revision 7 (package reconciliation):

- a negative RREL29, RREL67 or RREC13: P6 violation; a negative RREL30,
  RREL39, RREL73 or RREL74: no sign finding;
- RREC15 or RREL7 after the cut-off: a compliance violation; a maturity
  date or a lock-out end date before the cut-off: no violation;
- a cut-off date after the run date: violation;
- a lower-case originator name: an ITS format finding that does not
  count; a valid LEI pattern with wrong check digits cites ISO 17442;
- RREL30 = 100 with an empty companion and RREL28 = EUR: currency
  stated, no compliance violation and no "unknown unit"; the same with
  RREL28 also empty or ND: compliance violation (6.2 case 4) under
  either answer to decision 12; companion `EURO`: its own format
  finding;
- a test that the catalogue's ND flags and code lists equal the
  package's for all 104 fields (the comparison done by hand here).

Added with revision 8:

- final report: RREL66 is never expected to be 0 (ND5 expected, a date
  is the question); RREL68 other than 0 is the question;
- final-inactive RREL29: ND5 expected; a value is an L1 question;
- final-inactive repurchase (REBR, RREL8 a date, RREL9 = ND5, RREL30 =
  0): no lifecycle finding; the same with RREL8 = ND5: one C27 question;
- PERF, RREL30 = -1,000, RREC13 = 10,000, RREC12 = 0: no E9 discrepancy;
- negative RREL40 with negative RREL30: context only; with positive
  RREL30: a question; never a violation;
- current valuation in EUR and original valuation in JPY: E8 not
  evaluable with the reason; no A17 unless the JPY valuation was
  selected;
- RREL28 present and the RREL30 companion column absent: currency
  stated (case 2), a layout note, no compliance violation; both absent:
  violation;
- 100 retained loans with a negative RREL29: compliance violation on
  100, and rule 2 met or not met exactly as assumption h is decided;
- RREL23 after the cut-off seen by P6(b) and C3: one defect in rule 4;
- cut-off yesterday, today and tomorrow against a fixed assessment
  date: pass, question, violation;
- rental-income column absent: nothing; supplied with `-5` or `ND1`:
  violations; supplied and valid: assessed and shown as an optional
  element;
- gross principal 100 and savings 80 reported as RREL30 = 20 and RREL77
  = 80: unusual at most, with the Q&A 1343 wording; no cap violation.

Added with revision 9 (closing checks only):

- a repurchase date on PERF: one question from C10(a2), nothing counted;
  a redemption date on PERF: C10(a1) violation;
- RREL51 on or before the cut-off: one finding from C13, basis X, not
  counted, and nowhere described as a workbook rule;
- RREC13 = -1 with RREC17 = 100: P6 violation, no E3 exclusion reported;
- cut-off tomorrow: A10(a) reported once, nothing added to rules 2 or 4;
- FLCF with RREL47 = 12 and RREL51 = ND5: C38 and C12(c) violations;
- a loan with no collateral record: compliance violation plus the
  rule 3 loss, counted once.

Resource figures are targets until measured on the large generated case:
peak memory (raw text, parsed values, masks, relationship tables,
pipeline frames and export buffers can coexist), detail-store size, peak
disk use, export time, and detail-query latency.

The frontend has no test runner today; adding one is a dependency
decision for the frontend PR's brief, not assumed here.

### 11.3 PR split

Each PR leaves `main` usable; rejection behaviour reaches users only when
every consumer can represent it.

| PR | Content | Behaviour change |
|---|---|---|
| 1 | Parity gate fix (10.5) first. Then the stage interface (9.1): exposed predicates, decision results, `PipelineDataError`, the catalogue of expected fault sites. | none |
| 2 | Audit read, value states, catalogue with sources and reference lists, P rules, the two inventories, family A, populations and the waterfall, check outcomes, detail store, CSV, report workbook, summary JSON. CLI writes the report files beside the workbook. | additive: a report appears; the pipeline's success and failure are exactly as today |
| 3 | Families C, D, E, F, G; conditional ND5 rules; dropped-loans list complete across all exclusion points; lineage (P10) | additive |
| 4 | The decision: Stops, the four rules, override; the result contract; worker, handlers, schemas, CLI exit codes; `docs/api.md`; and the frontend section, in the same PR so no consumer sees an outcome it cannot show | rejection and report-only outcomes go live |
| 5 | `mapping-reference.html` Part D, `docs/analysis.md`, deployment notes; anything deferred | docs |

The waterfall observes steps 1 to 5 (Stage 2) and, from the stage
results of PR 1 and the links, every loss after Stage 2 with its cause:
no line is "cause detail pending" (8.2). While PRs 2 and 3
are additive the report never says MILAN was blocked or rejected: it
shows "policy outcome (not enforced in this release)" beside a MILAN
file that was produced under today's behaviour.
If PR 4 is too large to review, the API half can merge behind a setting
that keeps the old response until the frontend half merges; that is for
the brief.

As delivered, the second PR held the catalogue and the reference lists,
the audit read, the value states, the P rules, the two inventories, the
population, identity and lifecycle labels, the check outcomes and the
summary JSON written by the command line. The PR after it held the
detail store, the links, family A, the waterfall with the dropped-loans
list, G8 and A17, the detail CSV, the dropped-loans CSV and the report
workbook, and the conditional ND5 rule through the link. Still to come:
families C, D, E, F and the rest of G, L1, lineage (P10), the sheets
appended to the MILAN workbook, and everything of PR 4.

---

## 12. Dependencies

| Item | State | Effect |
|---|---|---|
| #45 identifier collision | open | A12 Stop until merged. Confirmed in the research for PR #49: the R reference recomputes the property identifier set after the intersection (`r_reference/R/pipeline.R:283-300`) while Python keeps the earlier set, so the fix is a parity correction, not a deviation. |
| #46 trim identifiers | open | A13a Stop until merged; a deviation as recorded |
| #47 Recourse `true`/`false` | open | reported as mapper compatibility (L) |
| New: boolean interpretation | not filed | Residency and Recourse depend on the column's inferred type and on different literals (6.2). One issue: a single interpretation rule for both fields, with today's behaviour described accurately first. |
| New: dated ND4 | not filed | `ND4-YYYY-MM-DD` is the only compliant ND4 form; the loader raises on it in numeric fields, the date parser nulls it with a warning, text fields keep it. Recommend recognising it as missing. |
| New: `NA` country code | not filed | the loader's global `NA` token turns the literal value `NA` (Namibia) into missing in RREL81 and RREL84. `NAZZZ` in a region field is not affected. |
| New: parity comparison cap | not filed | 10.5 |
| New: loader strictness on inactive records | not filed | an unparseable number on a redeemed loan blocks the whole pool (A5). Stays a Stop unless the owner wants the loader changed. |
| Synthetic fixture codes | merged (PR #48) | see 11.2 |
| `docs/parity-protocol.md` Rscript paths | noticed | outside this work |

---

## 14. Thresholds (config.py constants)

Each constant carries its basis (A or P) and a one-line reason.

| Constant | Value | Used by |
|---|---|---|
| `DQ_REJECT_CRITICAL_SHARE` | 0.10 | rule 2 |
| `DQ_REJECT_DROPPED_BALANCE_SHARE` | 0.05 | rule 3 |
| `DQ_REJECT_MANY_CRITICAL_COUNT`, `_SHARE` | 5, 0.01 | rule 4 |
| (removed) `DQ_ZERO_BALANCE_ACTIVE_SHARE` | | the count-based measure is undecided (question 3); no constant until it is |
| `DQ_CRITICAL_FIELDS`, `DQ_CRITICAL_CHECKS`, `DQ_CRITICAL_P_RULES` | section 8.3 | criticality |
| `DQ_DATE_MIN_YEAR`, `DQ_DATE_MAX_YEARS_AFTER_CUTOFF` | 1950, 60 | P12 |
| `DQ_MATURITY_MAX_YEARS_AFTER_CUTOFF` | 50 | C3 |
| `DQ_TERM_TOLERANCE_MONTHS`, `DQ_PAYMENTS_TOLERANCE` | 3, 3 | C5, C26 |
| `DQ_INCOME_MIN`, `DQ_INCOME_MAX` | 1,000; 10,000,000 | C20 |
| `DQ_RATE_MAX` | 20 | C11 |
| `DQ_DTI_SCALE_SUSPECT_MEDIAN` | 20 | C21 (pool-level scale suspicion) |
| `DQ_DTI_RECOMPUTE_FACTOR` | 2 | C47 (a separate check: single-loan recomputation note) |
| `DQ_LTV_TOLERANCE_POINTS` | 5 | E9 |
| `DQ_VALUATION_RATIO_MIN`, `_MAX` | 0.3; 5 | E8 |
| `DQ_COPIED_VALUATION_SHARE` | 0.90 | E7, C33 |
| `DQ_SAME_ORIGINATION_DATE_SHARE` | 0.50 | C32 |
| `DQ_ANNUITY_FACTOR` | 3 | C37 |
| `DQ_BORROWER_COMPARE_MONTHS` | 12 | D1 |
| `DQ_MAX_LOANS_PER_BORROWER` | 20 | D3 |
| `DQ_MAX_GROUP_LOANS`, `DQ_MAX_GROUP_PROPERTIES` | 20, 20 | F7 |
| `DQ_ND_SHARE_WARN` | 0.95 | lineage warning |
| `DQ_WA_RATE_BAND`, `DQ_WA_LTV_BAND`, `DQ_WA_SEASONING_BAND` | (0.5, 15), (0.1, 1.5), (0, 40) | G2 |
| `DQ_BORROWER_CONCENTRATION`, `DQ_TOP20_CONCENTRATION` | 0.05, 0.50 | G3 |
| `DQ_INVALID_MISSING_MARKERS` | list in 6.1 | P3d |
| `DQ_POSSIBLE_PLACEHOLDERS` | per-field dict | P8 |
| `DQ_ROWS_PAGE_MAX` | 200 | API |
| `DQ_XLSX_MAX_DETAIL_ROWS` | 500,000 (measured: 10.3) | 10.3 |
| `DQ_A12_IS_STOP`, `DQ_A13_IS_STOP` | True until #45, #46 merge | A12, A13 |

---

### 15.1 Decision record (signed 16 November 2026)

the owner decided each of the 18 items below on 16 November 2026, one by
one. Sixteen follow the proposal; decisions 3 and 7 differ from it and
are marked. A deferral with stated behaviour is a decision. This table
supersedes the lists under 15.2, which are kept as background only.

| # | Item (old reference) | Decision |
|---|---|---|
| 1 | Page layout (a) | Four tiles; the policy tile shows all six states in plain words. |
| 2 | C1 and C11 (b) | Stay plausibility; never count. |
| 3 | Mixed currencies in pipeline arithmetic (c) | **Changed by the owner:** processed with a prominent warning (A17, Warn), not blocked. The report names the affected calculations, loans and MILAN columns (8.5). |
| 4 | Exact duplicate headers (d) | Stop. |
| 5 | A share that straddles a threshold (e) | "Not determined": withheld unless overridden; the report keeps saying not determined. |
| 6 | Reference lists (f; question 2) | Bundled, versioned data: ISO 3166-1, ISO 4217 with historic codes, and NUTS 2010, 2013, 2016, 2021 and 2024. Any other year gets "limited assessment", never validation against a different year. |
| 7 | Excluded collateral record on a surviving loan (g) | **Changed by the owner:** counts only when the loan is left with no usable collateral record. A surviving loan's excluded record is a warning that does not count (3.2). |
| 8 | Published value rules, P6 (h) | Only the listed cases count: an amount below 0 in RREL29, RREL67, RREC13, RREC17; a date after the cut-off in RREL23, RREL66, RREC15, RREC19. |
| 9 | RREC3 (question 7) | Critical. |
| 10 | Over-long text on a critical field (question 9) | P7(a) length counts; P7(b) capitals and ASCII never counts. |
| 11 | Selected valuation without method or date (question 4) | First release: reported, no extra rejection condition. Omitted values under E5(a) count as already specified. |
| 12 | Genuinely unstated currency (question 8) | Processed with a Warn under a printed single-currency assumption (the currency, its scope and its basis), after the dedicated-field evidence of 6.2; the compliance violation stays; a known conflict is never overridden and stays visible as the A17 warning or the C30 conflict. |
| 13 | Count-based measure for loans lost before Stage 2's output (question 3) | Deferred. First release reports the four populations and their balances and enables no new threshold. |
| 14 | Synchronous calls (question 10) | Summary only when release is not permitted; the job route and the CLI give the full row-level report. |
| 15 | Lifecycle presumption (question 6) | Adopted as a labelled operational presumption from status and event dates, with the field map of 4.3 and its stated uncertainty. |
| 16 | Pipeline issues (question 5) | Tracked as separate issues (boolean interpretation, dated ND4, the `NA` country code, loader strictness on inactive records). Only the parity comparison cap is a prerequisite, fixed in PR 1. |
| 17 | Q&A sweep (question 1) | Done (4.4); its limitations stay on the report. |
| 18 | Implementation detail | The detail-store technology, the frontend test runner, measured resource limits and serialisation are settled in the PR briefs under this plan, without another planning round. |

The next step is the PR 1 brief (11.3); the source mapping file (4.3)
becomes the catalogue it is reviewed against.

## Appendix A: mapping from earlier check ids

| Earlier | Now |
|---|---|
| B1, B2, B3 | P5 and P12 on dates (format and plausibility separated) |
| B4, B5 | P5 on LIST and Y/N fields |
| B6 | P3a, P3b; P3c to P3g are new |
| B7 | P12 (suspicion only) |
| B8 | P6, only where the definition supports a sign rule |
| B9, B10 | P5, P11 |
| B11 | C30 and section 8.5 |
| B12 | field rule on RREL65 |
| B13 | P2 |
| C29, D4, E14 | retired |
| C12, C19, C21, C26, C27, C28, C43, C46, E9, E10, E17, F3, G6, G8 | rewritten (see Revisions) |
| G5 | lineage warning (9.3) |
| new | P11, P12, A13b, A16, C47, populations and waterfall |

## Appendix B: MILAN output lineage by ESMA field

From `pipeline/milan_map.py` at `9cfd584`. Class as in 9.3.

| ESMA field | MILAN output | Class | When the input is missing |
|---|---|---|---|
| RREL2, RREL3 | Loan Identifier, Additional data 1, 21, 22 | Gate | Stage 3 fault or loan excluded |
| RREL4, RREL5 | borrower identifier generation; Borrower Identifier, Additional data 2, 23, 24 | Gate, Direct | an absent RREL4 column can fail Stage 3 |
| RREL8, RREL9 | Repurchase Date, Redemption Date | Direct | "ND" |
| RREL10 | Borrower Residency | Direct | "ND"; a value reads `Y`/`N` only when it reaches the mapper as `TRUE`/`FALSE` (depends on the column's inferred type) |
| RREL11 | Borrower Postal/Zip Code | Direct | "ND" |
| RREL13, RREL17 | Employment Type; Borrower Type | Direct; Conditional | Employment Type "ND". Borrower Type is "2" when either input says legal entity, "1" when employment is present and not NOEM, so one missing input does not always give "ND" |
| RREL16, RREL20 | Income Borrower 1 and 2; Total Income | Direct; Conditional | "ND" when both are missing |
| RREL19, RREL21 | Income Verification (primary, secondary) | Direct | "ND" |
| RREL22 | Social Programme Type | Default | "6"; recognised scheme text "1" to "4"; other text "5" |
| RREL23, RREL24 | Origination Date (and seasoning), Maturity Date | Direct | RREL23 is required |
| RREL26, RREL27 | Origination Channel, Loan Purpose | Direct | "ND" |
| RREL29 | Loan OB | Direct | "ND" |
| RREL30 | Loan CB; Loan Currency (companion) | Gate | loan excluded |
| RREL31 | Total Prior Ranks CB, Ranking, External Prior Ranks CB | Group-derived | External Prior Ranks "ND" |
| RREL32 | Pari Passu Ranking Loans (Not In Pool) | Group-derived, Default | "0" |
| RREL33 | Flexible Loan Amount | Default | "0"; floored at "0" |
| RREL35 | Principal Payment Type | Direct | "ND"; OTHR "ND" |
| RREL37, RREL38 | Principal, Interest Payment Frequency | Direct | "ND" |
| RREL39, RREL40 | Periodic Payment, DTI Ratio | Direct | "ND" |
| RREL42 | Interest Rate Type | Direct | "ND" |
| RREL43, RREL46, RREL50 | Interest Rate, Full Margin, Margin After Reset Date | Direct | "ND" |
| RREL44, RREL45 | Base Index | Conditional | "ND"; LIBO or EURI without tenor "ND" |
| RREL51 | Interest Reset Date | Direct | "ND" |
| RREL58 | Number of Payments Before Securitisation | Direct | "ND" |
| RREL65, RREL69 | Restructured Loan, Restructuring Date | Conditional | "ND" when both missing |
| RREL66, RREL68 | Months Current, Months In Arrears | Conditional | "ND" when days missing |
| RREL67 | Arrears Amount | Direct | "ND" |
| RREL69 | Account Status | Gate, Direct | loan excluded; OTHR "ND" |
| RREL70 to RREL74 | default reason, amount, date, losses, recoveries | Direct | "ND" |
| RREL76, RREL77, RREL78 | Recourse, Deposit Amount, Additional Collateral Provider | Direct | "ND" (Recourse also for any spelling but `Y`/`N`) |
| RREL82 | Originator Identifier | Direct | "ND" |
| RREC2, RREC4 | the join; Property Identifier, Additional data 3, 25 | Gate | record or loan excluded |
| RREC3 | candidate property identifier (Stage 3); Additional data 26 | Gate, Direct | can fail identifier generation when the column is absent |
| RREC5, RREC9 | record kept or not; Property Type | Gate, Direct | record excluded |
| RREC6, RREC7 | Property region, Occupancy Type (main property) | Direct | "ND" |
| RREC8 | Additional data 14 | Direct | "ND" |
| RREC10, RREC11 | EPC value, EPC provider | Direct | "ND" |
| RREC12, RREC16 | Current LTV, Original LTV | Direct | "ND" |
| RREC13 to RREC15, RREC17 to RREC19 | Property Value, Property Valuation Type and Date; Additional data 10 to 12, 17 to 20 | Gate, Selected, Group-derived | record excluded when no valuation above 0; a missing valuation column can fail Stage 2 or Stage 7 before the mapper pads anything |
| RREC21 | Sale Price | Direct | "ND" |
| RREC23 | MIG Provider; Type of Guarantee Provider | Default; Direct | "No Guarantor"; "ND" |
| none | Property 2 to 5 blocks, Servicer Identifier, the literal "ND" columns, Retained Amount | Constant | not caused by any input |

In that verification "missing" is split three ways, because they behave
differently: the column absent from the file, a null cell (empty or a
token the loader nulls), and a permitted ND value that reaches a text
comparison. A missing value in the loan-identifier column the pipeline
did not select does not by itself exclude the loan.

The "when missing" column is to be confirmed cell by cell against the
mapper in the PR 3 research round; the statement "an absent mapped column
always reads ND" from the first version was too broad and is withdrawn.

---

