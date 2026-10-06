# Public Project Release Audit

## Scope and status

This package is the Typed Decision Reliability research project: Phase 1 MCQA,
Phase 2 training/calibration/robustness/transfer/live cascade, and the separate
MacroLab module. It is a new publication tree with no inherited private history.
The original experiments and earlier figure-only preparation remain untouched.
These checks were completed before publication. The user subsequently authorized
publishing only this sanitized package to the specified GitHub repository.

## Included evidence and checks

- 78 original method Python snapshots, plus portable figure scripts and release tests/tools.
- 27 aggregate CSV exports, including the historical freeze matrix and citation map.
- Every row retained in each selected table; retained cell values match the stored source strings exactly.
- Public settings contain hyperparameters and counts, not dataset IDs or question content.
- Source and publication SHA-256 hashes and explicit transformations are indexed in SOURCE_MAP.csv.
- Original included sources were hash-checked again after packaging; no source writes were performed.
- Method Python ASTs match their originals except for the documented private-seed lookup in one execution file.
- The two PNG/PDF figures are unchanged from the previously validated public color figure package.
- All reviewed text is scanned for common credential formats, private keys, JWT/bearer tokens, contact information, personal absolute paths and actual case IDs.
- JSON/config schemas are checked for forbidden private-data fields; PDF text/metadata and PNG metadata are checked separately.
- Synthetic public-schema and bounded-interpreter tests use no model or experiment data.
- Git stages only an explicit file allowlist; ignore rules are tested against representative sensitive paths.
- The ZIP is built from that allowlist and verified against the staged-file hashes, without .git or private audit materials.

## Publication-only transformations

Text is packaged as UTF-8/LF. Relative evidence-pointer columns are removed from
aggregate tables; every other stored value and all rows remain. Local citation
source paths are removed. Configuration exports replace data-ID lists with counts
and omit host/model path maps, private task seeds and private reconstruction data.
An already superseded temperature descriptor is explicitly omitted; the original
authoritative calibration-view note is retained. Named MacroLab case identifiers
in the historical ledger are replaced without changing the result counts.

The Phase 1 summary excerpt removes case-specific answer examples and explicitly
labels its historical AI-review scope. Aggregate negative results remain.
The execution snapshot reads its signature seed from a private seed file rather
than containing the original hidden-task seed. This is a publication adaptation,
not an experiment modification, and does not certify the original frozen hashes
for the adapted file. No new frozen protocol is created.

## Excluded materials

Credentials, personal documents/contact information, raw requests/responses/traces,
question content/IDs, gold and hidden arrays/targets, individual predictions,
private MacroLab generator/oracle, exact split-reconstruction scripts, original
locks containing IDs, weight/checkpoint/optimizer/RNG files, logs, environments,
caches and original large archives are withheld. Case-specific report generators
and review packets are not included. EXCLUDED_CODE.json lists excluded source files.
The previous interactive-toolmaking experiment remains outside this release.

## Limitations

The checks cover this exact reviewed release, not every source file in the research
workspace, and pattern scanning cannot prove absence of every possible secret
encoding. No model, original experiment or environment installation is retested.
The archive requires private evidence and dependencies for exact original-study
reconstruction; the runnable release checks and figure scripts are separate.
Historical reports, AI-review flags and imported citation records retain their
original scopes. No human-review or new online citation-verification claim is added.
Publication identity uses the owner's GitHub username and noreply address; a
license has not been invented. The destination repository has been provided for
the authorized push. Forced Git adds can bypass ignore rules.
