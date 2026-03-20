# Manuscript Patch List

Tracks all required revisions to RESEARCH_PAPER.md.

## Status Summary (2026-03-19)

ALL 13 PATCHES APPLIED to docs/RESEARCH_PAPER.md and docs/RESEARCH_PAPER.tex. PDF recompiled.

| Patch | Description | Status |
|-------|-------------|--------|
| P001 | Replace proxy mortality with real outcomes | APPLIED |
| P002 | Reframe "dynamic subtype" language | APPLIED |
| P003 | Reduce cross-center generalization claim | APPLIED (superseded by P013) |
| P004 | Flag Sepsis 2019 silhouette as unresolved | APPLIED |
| P005 | Document outcomes file usage | APPLIED |
| P006 | Add representation comparison | APPLIED |
| P007 | Update references to 2025 | APPLIED |
| P008 | Add center probe results | APPLIED |
| P009 | Add representation comparison table | APPLIED |
| P010 | Add S2-light temporal trajectory section | APPLIED |
| P011 | Add phenotype prevalence shift | APPLIED |
| P012 | Add stride sensitivity supplementary note | APPLIED |
| P013 | S3 cross-center validation with calibrated wording | APPLIED + VERIFIED |

## P014 — Final submission-ready audit polish (2026-03-20)
- **Scope:** Full scientific, numeric, figure/table, reference, and claim-calibration audit
- **Edits applied:**
  1. Abstract: "2~independent centers" → "2~centers"
  2. Introduction 1.3: Added 6th contribution bullet (S3 cross-center validation)
  3. Methods 3.7: Added software tool citations (scikit-learn, matplotlib)
  4. Results 4.5 + Discussion 5.3 + Future 5.4: Added \cite{pollard2018} for eICU-CRD references
  5. Supplementary Fig S5 caption: Added proxy-label provenance warning
  6. Bibliography: Removed unused UMAP reference (mcinnes2018)
  7. Bibliography: Fixed zhang2024dtw first author (was missing, now "H.~Zhang")
- **Verification:** splits.json confirms encoder trained exclusively on Center A (6,191 train patients, 0 from Center B)
- **Status:** APPLIED

## Remaining Items

- None — all patches applied, PDF compiled, wording verified

## P013 — Add S3 cross-center validation section with calibrated wording
- **Location:** Abstract, new Section 4.5, Section 5, Section 6
- **Old claim (P003):** "preliminary cross-center consistency"
- **New claim:** "cross-center temporal validation within the PhysioNet 2012 multi-center cohort, with all six validation criteria passed including identical mortality ordering across centers"
- **Language rules:**
  - USE: "cross-center temporal validation within the PhysioNet 2012 multi-center cohort"
  - USE: "out-of-center validation (train on Center A, evaluate on Center B)"
  - DO NOT USE: "robust cross-center generalization"
  - DO NOT USE: "independent hospital generalization"
- **Required caveat:** "Both centers derive from the same PhysioNet 2012 database. Full external validation requires independently collected ICU cohorts."
- **Evidence:** data/s3/cross_center_report.json, E015, D015
- **Status:** APPLIED (2026-03-19)

## P001 — Replace all proxy mortality numbers with ground truth
- **Location:** Abstract, Section 4.1.2, Section 5.1, Section 6
- **Current text:** "mortality profiles (27.9%–58.3%, p < 0.001)"
- **Required change:** Replace with real mortality rates from S0 outcomes (total rate ~14.2%, range varies by clustering run)
- **Source:** data/s0/outcome_audit_report.json, data/s0/clustering_consistency_report.json
- **Status:** NOT YET APPLIED

## P002 — Correct "dynamic subtype" language
- **Location:** Title, Abstract, throughout
- **Current text:** Title says "Dynamic Subtype Discovery"
- **Required change:** The current system performs static 48h-window clustering. "Dynamic" must either be removed or qualified as "toward dynamic subtyping" until S2 implements true temporal phenotyping.
- **Status:** NOT YET APPLIED

## P003 — Reduce "robust cross-center generalization" claims
- **Location:** Abstract, Section 4.1.4, Section 6
- **Current text:** "robust cross-center generalization"
- **Required change:** Downgrade to "preliminary cross-center consistency" — current evidence shows matching subtype proportions, not strict train-A/test-B validation.
- **Status:** NOT YET APPLIED

## P004 — Mark Sepsis 2019 silhouette inconsistency
- **Location:** Section 4.2.1
- **Current text:** Reports silhouette=0.435 for simulated data at K=2
- **Required change:** The Sepsis 2019 silhouette values are UNRESOLVED pending independent rerun. Must be explicitly flagged or removed until verified.
- **Status:** UNRESOLVED — pending rerun

## P005 — Update data section with S0 improvements
- **Location:** Section 3.2, Section 3.4
- **Required change:** Document that mortality labels now come from Outcomes files (not proxy). Document observation mask preservation. Document proxy vs intervention separation.
- **Status:** NOT YET APPLIED

## P006 — Add self-supervised representation results (when S1.5 complete)
- **Location:** Section 3.6, Section 4
- **Required change:** Replace PCA-only representation with comparison table (PCA vs SS encoder). Update architecture description.
- **Status:** BLOCKED on S1.5 completion

## P008 — Add corrected center probe results
- **Location:** New section or supplementary
- **Required change:** Report that center probe AUROC ≈ 0.50 for all representations, confirming no center leakage. Document the probe methodology (random stratified split, not cross-center phenotyping split).
- **Status:** NOT YET APPLIED

## P009 — Report representation comparison results
- **Location:** Section 3.6 (Representation Learning), Section 4 (Results)
- **Required change:** Replace PCA-only results with 4-way comparison table (PCA / S1 masked / S1.5 contrastive / S1.6 ablation). Document that S1.5 was selected for downstream analysis based on weighted multi-criteria evaluation (D010).
- **Caveat:** PCA still leads K=4 mortality separation. This must be reported honestly, not hidden.
- **Status:** NOT YET APPLIED

## P010 — Add S2-light temporal phenotype trajectory section
- **Location:** New Section 4.x or Section 5 (Results/Discussion)
- **Required content:**
  - Rolling-window method description (24h windows, 6h stride, frozen S1.5 encoder)
  - Per-window cluster prevalence figure (per_window_prevalence.png)
  - Transition flow diagram (sankey_transitions.png)
  - Transition statistics: 64.8% stable, 29.3% single-transition, 5.9% multi-transition
  - Mortality by stable phenotype: P0=4.0%, P1=22.5%, P2=31.7%, P3=9.7%
  - Mortality by trajectory category: stable=15.4%, single-transition=11.4%, multi-transition=15.2%
- **Language constraint:** Must use "descriptive temporal phenotype trajectories" not "dynamic phenotyping" or "dynamic subtype discovery." The title word "dynamic" must be qualified or replaced.
- **Status:** NOT YET APPLIED

## P011 — Phenotype prevalence shift over time
- **Location:** Results section
- **Required content:** Phenotype 0 grows from 24.6% (W0) to 33.1% (W4); Phenotype 1 shrinks from 35.7% to 28.1%. This suggests a population-level drift toward Phenotype 0 (low-risk) over the 48h window, consistent with clinical stabilization.
- **Caveat:** This is descriptive, not causal. The shift may reflect treatment response, survivor bias, or measurement pattern changes.
- **Status:** NOT YET APPLIED

## P012 — Add stride sensitivity supplementary note
- **Location:** Results section (supplementary or inline footnote)
- **Required content:** "Sensitivity analysis with reduced overlap (stride=12h, 50%) confirmed identical mortality stratification ordering and range (28.0pp vs 27.7pp). Transition rates were higher at reduced overlap (19.1% vs 10.4%), indicating the primary analysis provides a conservative estimate of temporal phenotype change."
- **Key finding:** Overlap suppresses transitions, not creates them. Stride=6h results are conservative.
- **Status:** NOT YET APPLIED

## P007 — Update reference list
- **Location:** References section
- **Required change:** Add 2025 references from field investigation (Antcliffe ICM, Zheng npj, Amirahmadi JMIR, Feng EClinicalMedicine, etc.)
- **Status:** NOT YET APPLIED
