# Next Steps

## Current Position

Manuscript is **SUBMISSION-READY WITH MINOR NON-BLOCKING CAVEATS**. Full scientific audit completed 2026-03-20. All 14 patches (P001-P014) applied and verified. PDF compiled (17 pages, 1.3MB, 0 errors). 20 references all cited. No prohibited phrases. All numeric cross-references verified.

2026-03-23 update: supplementary downstream mortality validation on frozen S1.5 embeddings was added to the repo and manuscript, and the paper PDF was recompiled.

2026-03-24 update: Stage 4 / Stage 5 follow-up research framework is now implemented in code. The repo contains treatment-feature extraction for MIMIC/eICU, a treatment-aware S1.5 fusion model, causal analysis utilities (PSM / DML causal-forest-style / RDD), a distilled realtime student, note-embedding utilities, and a bedside HTML dashboard prototype. What remains is full-dataset execution and real clinical validation, not framework implementation.

2026-03-31 update: downstream calibrated stacking was expanded from a local neighborhood search to a branch-specific combo search over `900` calibration candidates. The winning configuration uses `stats_hgb_d5_lr0.03_iter200`, `fused_hgb_d5_lr0.03_iter200`, `meta_C=0.02`, and post-hoc `temperature` scaling. Updated Center B calibrated metrics are `Brier=0.0895`, `ECE=0.0198`, `AUROC=0.8731`, `balanced_accuracy=0.7937`, `recall=0.8376`, and `precision=0.3646` at threshold `0.09`.

2026-04-01 update: Stage 4 full-cohort external execution is complete on both MIMIC-IV and eICU. Reproducible S4 closeout and S5 validation scripts were executed successfully and generated stable reports/figures under `outputs/reports/s4/`, `outputs/reports/s5/`, and `docs/figures/`. A same-day cloud rerun of S5 also completed on both MIMIC-IV and eICU and was pulled back into `data/s5_cloud/` and `outputs/reports/s5_cloud_20260401/`. S4 shows `0/6` cross-source recommendation consistency, while S5 passes all predefined engineering gates on both external sources in both the local packaged reports and the cloud rerun. A full S5-v2 evaluation was then completed on the real external cohorts: the causal-TCN branch was not promoted after real-data regression on MIMIC-IV, while a calibrated transformer-based S5-v2 path with richer distillation and post-hoc temperature scaling passed all predefined gates on both MIMIC-IV and eICU and now serves as the preferred Stage 5 engineering candidate.

2026-04-02 update: bedside silent deployment / prospective-style hourly replay is now implemented and executed on the frozen calibrated S5-v2 artifacts for both MIMIC-IV and eICU under `outputs/reports/s5_silent_20260401/`. The replay code produces patient-level summaries, landmark metrics, cumulative alert curves, and a sample bedside dashboard. First deployment findings are mixed: eICU is plausible but still alert-heavy, while MIMIC-IV becomes operationally non-viable at the current final-window threshold (`patient_alert_rate≈0.994`). The blocker is now deployment-policy tuning, not another student-architecture search.

2026-04-06 update (Part 1 — policy tightening): MIMIC-IV deployment policy tightened to shadow-ready. Root cause: previous searches used wrong bundle and omitted min_history=7h. Best shadow policy: thr=0.87, hist=7h.

2026-04-06 update (Part 2 — model fine-tuning): MIMIC-IV S5 model fine-tuned with horizon augmentation. Root cause of over-prediction identified as training-deployment distribution mismatch (full-sequence training vs partial-sequence deployment). Fix: randomly truncate sequences to random horizon h∈[6,48] during fine-tuning. Result: negative patient risk at h6 drops from 0.675 → 0.362; 276 PRODUCTION-feasible policies found. Best production policy: thr=0.75, hist=8h, neg_alert=0.130, pos_alert=0.624, pos@24h=0.503. MIMIC-IV status: `shadow_only` → `production_ready`. Model: `data/s5_mimic_finetune_horizon_aug_20260406/realtime_student.pt`.

## Completed Stages

- S0: Data layer refactor + real outcomes (DONE)
- S1: Masked reconstruction encoder (DONE)
- S1.5: Masked + contrastive encoder (DONE)
- S1.6: Lambda ablation (DONE, did not beat S1.5)
- S2-light: Temporal phenotype trajectories (DONE, stride sensitivity validated)
- S3: Cross-center temporal validation (DONE, 6/6 criteria, calibrated wording)
- S3.5: Downstream mortality validation on frozen S1.5 embeddings (DONE, Center B AUROC 0.829 / balanced accuracy 0.745)
- S3.5-calibration: Branch-specific calibrated stacking optimization + comparison (DONE, Center B Brier 0.0895 / ECE 0.0198 / AUROC 0.8731)
- S4: Full-cohort external treatment-aware execution + observational causal analysis (DONE, MIMIC-IV and eICU)
- S4-closeout: Reproducible summary tables/figures from completed S4 artifacts (DONE)
- S5-engineering-validation: Realtime student artifact validation summary on MIMIC/eICU reports (DONE)
- S5-v2-calibrated-transformer: Real-cohort S5-v2 closeout on MIMIC/eICU with post-hoc temperature scaling (DONE)
- S5-silent-deployment-replay: Hourly bedside replay and prospective-style validation on frozen S5-v2 artifacts (DONE)
- S5-mimic-deployment-policy-tightening: MIMIC-IV deployment policy tightened to shadow-ready (DONE, 2026-04-06, config/s5_mimic_deployment_policy.json)
- Manuscript: All 14 patches applied + audited (P001-P014), LaTeX compiled, submission audit passed

## Immediate Next Step

- S5 MIMIC-IV is now PRODUCTION-READY: fine-tuned model with horizon augmentation achieves neg_alert=0.130, pos_alert=0.624, aepd=0.109.
- Both MIMIC-IV and eICU have valid production deployment policies.
- Next meaningful work options:
  1. S5 prospective validation: run the production policy in true prospective mode on held-out future data
  2. S4 stronger causal identification: justified next S4 increment is stronger IV/RDD, not another observational baseline rerun
  3. Manuscript submission: paper is already submission-ready; fine-tuning results are follow-up research extensions

## Non-Blocking Caveats (do not prevent submission)

1. Supplementary survival curves (Fig S5) use V1 pipeline with proxy labels — noted in caption
2. PhysioNet/CinC 2019 Sepsis metrics flagged as unresolved in Section 4.6
3. Simulated data validation (ARI=0.245) is modest but honestly reported
4. 1 cosmetic overfull hbox in Table 1

## Available Future Stages

- S4: Treatment heterogeneity
  External execution is complete; remaining work is stronger causal identification and publication-quality interpretation, not another baseline rerun
- S5: Bedside real-time classifier
  Distilled student + dashboard prototype implemented in `s5/`; repo-level validation summaries are available, the calibrated transformer-based S5-v2 path is now validated on both MIMIC/eICU and preferred over the exploratory causal-TCN branch, and first bedside silent-deployment replay has been executed on both sources. What remains is deployment-policy tuning because the current MIMIC-IV alert burden is operationally too high.
- External database validation
  Full external temporal transfer is done; treatment-aware and note-aware full-dataset reruns remain pending
- Updated supplementary survival curves with S1.5 temporal phenotypes
