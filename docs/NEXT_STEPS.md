# Next Steps

## Current Position

Manuscript is **SUBMISSION-READY WITH MINOR NON-BLOCKING CAVEATS**. Full scientific audit completed 2026-03-20. All 14 patches (P001-P014) applied and verified. PDF compiled (17 pages, 1.3MB, 0 errors). 20 references all cited. No prohibited phrases. All numeric cross-references verified.

2026-03-23 update: supplementary downstream mortality validation on frozen S1.5 embeddings was added to the repo and manuscript, and the paper PDF was recompiled.

## Completed Stages

- S0: Data layer refactor + real outcomes (DONE)
- S1: Masked reconstruction encoder (DONE)
- S1.5: Masked + contrastive encoder (DONE)
- S1.6: Lambda ablation (DONE, did not beat S1.5)
- S2-light: Temporal phenotype trajectories (DONE, stride sensitivity validated)
- S3: Cross-center temporal validation (DONE, 6/6 criteria, calibrated wording)
- S3.5: Downstream mortality validation on frozen S1.5 embeddings (DONE, Center B AUROC 0.829 / balanced accuracy 0.745)
- Manuscript: All 14 patches applied + audited (P001-P014), LaTeX compiled, submission audit passed

## Non-Blocking Caveats (do not prevent submission)

1. Supplementary survival curves (Fig S5) use V1 pipeline with proxy labels — noted in caption
2. PhysioNet/CinC 2019 Sepsis metrics flagged as unresolved in Section 4.6
3. Simulated data validation (ARI=0.245) is modest but honestly reported
4. 1 cosmetic overfull hbox in Table 1

## Available Future Stages (not started)

- S4: Treatment heterogeneity (requires treatment variables; only proxies for PhysioNet 2012)
- S5: Bedside real-time classifier (early-window phenotype assignment; 48h frozen-embedding mortality prototype now exists)
- External database validation (requires eICU or MIMIC-IV access)
- Updated supplementary survival curves with S1.5 temporal phenotypes
