# Typed Decision Reliability

Research implementation and saved aggregate results for Typed Decision MCQA,
fine-tuning, probability calibration, option-order robustness, domain transfer,
live model routing, and the separate MacroLab exploration.

This is the **public project release preparation**, not just a figure package.
It contains the original method code snapshots from Phase 1 and Phase 2, public
views of the settings, aggregate tables, documentation, and the paper figures.
Private experimental evidence remains in the original local research workspace.
No training, inference, calibration fitting, or new experiment was run to prepare
this release. The user has authorized publishing this sanitized project package.

## Project layout

|Directory or file|Contents|
|---|---|
|`typed_decision_mcqa_15h/`|Phase 1 adapters, training, inference, scoring, analysis and validation code; public protocol view; five aggregate tables|
|`typed_decision_phase2/`|Phase 2 recipes, calibration, model adapters, robustness/transfer analysis, live cascade and audits; public settings; aggregate results|
|`typed_decision_phase2/macrolab/`|Bounded interpreter, typed episode policy, adapter, execution and evaluation source snapshots; public protocol view|
|`visualization/`|Two color paper figures, their minimal aggregate data and portable plotting scripts|
|`docs/`|Historical protocol summaries, findings, deviations, environment/model metadata and citation scope map|
|`SOURCE_MAP.csv`|Relative original source names, SHA-256 hashes and transformations|
|`PUBLIC_CSV_CHECKS.json`|Exact retained-value checks and row counts for every aggregate CSV export|
|`PUBLICATION_AUDIT.md`|Release checks, exclusions and limitations|
|`PUBLIC_FILES_SHA256.csv`|Integrity hashes for the reviewed publication files|

## Existing results

The historical E_old set and the new study-held E_new set are reported separately.
E_new is a public benchmark validation subset newly held out for this study; it
does not establish that the pretrained models never encountered these questions.

- The Phase 2 fine-tuned accuracy means are 48.61% (GRPO+CE) and 49.72% (CE-only),
  compared with Base 39.17% on the 240-question E_new set. A GRPO advantage over
  CE-only was not established.
- Fine-tuning worsened raw probability-quality metrics. The previously fitted
  shared calibration procedure improved them; probability views must not be mixed.
- On 235 rotation-safe questions, the s0 fine-tuned models had fewer answer flips
  than Base but greater aligned fitted probability drift.
- The live cascade used the locked bypass policy. It did not meet the planned
  20% mean-latency reduction target.
- In the fixed MacroLab exploration, both tested models had zero task successes
  in M0/M1/M2; the enumeration control achieved 30/40. The development library was
  empty, so the memory effect was not identifiable.

These are existing findings, not new release-time analyses. All rows of each
selected aggregate table are retained, including every applicable seed, fold,
view and negative result. Counts across different seeds, probability views or
rotations are not additional independent questions. See the historical summaries
in `docs/` and the complete selected CSVs for the applicable uncertainty definitions.

## Safe inspection and figure reproduction

The release-integrity validator and synthetic safety tests require only Python
3.10 or newer. They perform no model calls and use no experiment inputs:

```sh
python -B tools/validate_public_release.py
python -B tests/test_public_boundaries.py
```

To redraw the existing figures, install `requirements-figures.txt` in a separate
environment and run:

```sh
python -B visualization/scripts/make_fig1_calibration.py
python -B visualization/scripts/make_fig2_order_robustness.py
```

Those scripts read only the included aggregate plotting CSVs. They do not change
the reported numerical results. Running them may create ignored plotting caches.

## Experiment source and reproducibility boundary

The method snapshots preserve their original relative directory relationship.
`requirements-phase1-observed.txt` records the observed Phase 1 environment. The
Phase 2 environment reused that dependency base with a separate overlay; the
observed lock is not advertised as a complete portable Phase 2 installer.

The experiment runners are archived method code, **not one-click runnable studies**.
They require private data, original frozen locks, checkpoints and external model
runtime dependencies deliberately absent here. The `*_public.json` files are
documentation views, not replacements for the original runnable frozen configs.
Do not run orchestration scripts expecting them to recreate the private experiment.

Private split-reconstruction scripts and the MacroLab generator/oracle are
withheld. One public execution snapshot reads its signature seed from a private
`signature_seed.json` file instead of containing the original task seed. This
publication-only change is recorded in `SOURCE_MAP.csv`; it is not a change to
the original experiment or a newly frozen protocol.

Historical documents retain their original scope and may link to deliberately
withheld evidence. Historical AI-review flags are not a statement about every
later researcher review. Citation files are imported from the existing writer
support work, with their version/support limits; this release does not claim a
new online verification or manuscript-wide citation acceptance.

## Privacy and publication

The release omits credentials, private contact details, personal documents,
question content/IDs, gold answers, hidden inputs/outputs, private solvers,
requests, raw responses, per-question records, weights/checkpoints, training
logs, environments/caches and the original archives. The earlier paused
interactive-toolmaking experiment is outside this release.

The Git ignore policy denies everything except the exact reviewed file allowlist.
Do not force-add files or publish the original research archives. The public
repository starts with no original private Git history. Commit identity uses the
repository owner's GitHub username and noreply address. A publication license
has not been invented; no external third-party
model runtime source is redistributed. Dependency and model licenses remain
separate from this package.
