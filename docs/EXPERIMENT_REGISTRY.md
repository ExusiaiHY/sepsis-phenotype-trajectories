# Experiment Registry

All experiments with their configurations, results, and artifact locations.

## E001 — V1 PhysioNet 2012 baseline (original paper)
- **Date:** 2026-03-17
- **Method:** Statistical features (527) → PCA (32 dim) → KMeans (K=4)
- **Data:** PhysioNet 2012, 11,816 patients, proxy mortality labels
- **Results:** Silhouette=0.065, CH=765.3, DB=2.89, mortality range [27.9%, 58.3%]
- **Artifacts:** outputs/reports/evaluation_report.json, outputs/figures/
- **Issues:** Mortality labels are PROXY (PPV=17.7%). Results not trustworthy for absolute mortality rates.
- **Sepsis 2019 note:** Silhouette values from Sepsis 2019 dataset are UNRESOLVED pending rerun. Not carried forward.

## E002 — S0 clustering consistency check
- **Date:** 2026-03-18
- **Method:** S0 compat(exact_v1) → V1 feature engineering (436 features) → PCA (32 dim) → KMeans (K=2..6)
- **Data:** PhysioNet 2012, 11,986 patients, REAL mortality labels (14.2%)
- **Results:** K=4 silhouette=0.057, mortality range [6.4%, 37.0%] (range=30.6%)
- **Artifacts:** data/s0/clustering_consistency_report.json
- **Conclusion:** Directionally consistent with E001. Mortality rates differ due to label correction.

## E003 — S1 masked reconstruction pretraining
- **Date:** 2026-03-18
- **Method:** Transformer encoder (d=128, L=2, H=4), masked value prediction (15%), 50 epochs
- **Data:** S0 processed continuous (11,986 patients, 48h, 21 features), train split only
- **Results:** Best val_loss=0.209 (epoch 49), 296,085 parameters
- **Artifacts:** data/s1/checkpoints/pretrain_best.pt, data/s1/pretrain_log.json

## E004 — S1 vs PCA multi-seed clustering comparison
- **Date:** 2026-03-18
- **Method:** KMeans (5 seeds) on SS embeddings (128d) vs PCA embeddings (32d)
- **Data:** Full cohort (11,986 patients)
- **Results:**
  - K=2: SS sil=0.116±0.000, PCA sil=0.223±0.001; SS mort_range=14.4%, PCA mort_range=23.8%
  - K=4: SS sil=0.087±0.000, PCA sil=0.061±0.000; SS mort_range=17.6%, PCA mort_range=29.2%
  - Center stability: SS L1=0.024, PCA L1=0.027 (both low, comparable)
- **Artifacts:** data/s1/comparison_report.json, data/s1/embeddings_ss.npy, data/s1/embeddings_pca.npy
- **Conclusion:** SS improves geometric clustering (K=4 silhouette) but PCA produces stronger mortality separation. Neither is clearly dominant. S1.5 needed.

## E005 — S1.5 contrastive pretraining
- **Date:** 2026-03-18
- **Method:** Transformer encoder (d=128, L=2, H=4), masked reconstruction + NT-Xent contrastive on stochastic 30h windows, λ warmup 0→0.5 over 10 epochs, 50 epochs
- **Data:** S0 processed continuous (11,986 patients), train split only
- **Results:** Best val_loss=0.230 (epoch 47), 321,109 parameters. No collapse: cos_neg=-0.013, cos_pos=0.932, embedding_norm=8.8±0.37
- **Artifacts:** data/s15/checkpoints/, data/s15/pretrain_log.json

## E006 — 3-way clustering comparison (PCA vs S1 vs S1.5)
- **Date:** 2026-03-18
- **Method:** KMeans (5 seeds) on PCA (32d), S1-masked (128d), S1.5-contrastive (128d)
- **Results (K=4):**
  - PCA: sil=0.061, mort_range=29.2%, center_L1=0.027
  - S1 masked: sil=0.087, mort_range=17.6%, center_L1=0.024
  - S1.5 contrastive: sil=0.080, mort_range=24.6%, center_L1=0.016
- **Results (K=2):**
  - PCA: sil=0.223, mort_range=23.8%
  - S1 masked: sil=0.116, mort_range=14.4%
  - S1.5 contrastive: sil=0.084, mort_range=6.6%
- **Artifacts:** data/s15/comparison_report.json

## E007 — Representation diagnostics (3-way probes)
- **Date:** 2026-03-18
- **Method:** Linear probes (LogisticRegression/Ridge) on test split embeddings
- **Results:**
  - Mortality AUROC: PCA=0.825, S1=0.825, S1.5=0.830
  - LOS R²: PCA=0.102, S1=0.405, S1.5=0.343
  - Density-norm |r|: PCA=0.231, S1=0.247, S1.5=0.148
  - Center AUROC: NaN for all (center_b is entire test set; probe needs random-split variant)
- **Artifacts:** data/s15/diagnostics_*.json
- **Issues:** Center probe NaN due to single-center test split. Needs fix for center evaluation.

## E008 — Corrected center probe (all 4 methods)
- **Date:** 2026-03-18
- **Method:** LogisticRegression on random stratified 70/30 split (mixing both centers in train+test)
- **Data:** All 11,986 patients, center_a=7,989 / center_b=3,997
- **Results:**
  - PCA: 0.508, S1: 0.507, S1.5: 0.520, S1.6: 0.516
  - ALL near 0.50 — no method leaks center identity
- **Artifacts:** data/s16/diagnostics_*.json
- **Conclusion:** Center probe NaN was a split artifact. No center leakage in any representation.

## E009 — S1.6 pretraining (λ=0.2 ablation)
- **Date:** 2026-03-18
- **Method:** Same as S1.5 but max_lambda=0.2 (down from 0.5), 50 epochs
- **Data:** S0 processed, train split
- **Results:** Best val_loss=0.224, 321K params. No collapse.
- **Artifacts:** data/s16/checkpoints/, data/s16/pretrain_log.json

## E010 — 4-way comparison (PCA / S1 / S1.5 / S1.6)
- **Date:** 2026-03-18
- **Method:** KMeans (5 seeds) + diagnostics on all 4 embeddings
- **Results (K=4):**
  - PCA: sil=0.061, mort_range=29.2%, center_L1=0.027, mort_AUROC=0.825
  - S1: sil=0.087, mort_range=17.6%, center_L1=0.024, mort_AUROC=0.825
  - S1.5 (λ=0.5): sil=0.080, mort_range=24.6%, center_L1=0.016, mort_AUROC=0.830
  - S1.6 (λ=0.2): sil=0.079, mort_range=25.1%, center_L1=0.021, mort_AUROC=0.825
- **Artifacts:** data/s16/comparison_4way.json
- **Conclusion:** S1.6 does not improve over S1.5. S1.5 remains the best learned representation.

## E011 — S2-light rolling-window extraction
- **Date:** 2026-03-19
- **Method:** Frozen S1.5 encoder on 5 rolling windows (24h, stride 6h) per patient
- **Data:** S0 processed continuous, 11,986 patients
- **Results:** (11986, 5, 128) embeddings. Obs density decreases W0→W4 (0.279→0.254). ~160-195 sparse windows per position.
- **Artifacts:** data/s2/rolling_embeddings.npy, data/s2/rolling_meta.json

## E012 — S2-light temporal clustering
- **Date:** 2026-03-19
- **Method:** KMeans K=4, fit on train-split windows (30,955), assigned to all (59,930)
- **Results:**
  - Overall window-level silhouette: 0.077
  - Per-window silhouette range: 0.072 (W0) to 0.082 (W4)
  - All 4 clusters present at all 5 window positions
  - Phenotype 0 grows from 24.6% (W0) to 33.1% (W4); Phenotype 1 shrinks from 35.7% to 28.1%
- **Artifacts:** data/s2/window_labels.npy, data/s2/kmeans_model.json

## E013 — S2-light transition analysis
- **Date:** 2026-03-19
- **Method:** Empirical transition matrix from adjacent window pairs, patient-level trajectory classification
- **Results:**
  - Patient-level: 64.8% stable, 29.3% single-transition, 5.9% multi-transition
  - Event-level: 10.4% non-self transitions, entropy ratio 0.637
  - Top transitions: 1→0 (956), 1→3 (728), 3→0 (713)
  - Stable phenotype mortality: P0=4.0%, P1=22.5%, P2=31.7%, P3=9.7% (range 27.7pp)
  - Trajectory mortality: stable=15.4%, single-transition=11.4%, multi-transition=15.2%
- **Artifacts:** data/s2/transition_matrix.json, data/s2/trajectory_stats.json, data/s2/sanity_checks.json
- **Figures:** sankey_transitions.png, per_window_prevalence.png, mortality_by_trajectory.png
- **Conclusion:** Meaningful temporal structure detected. Not trivially stable. Phenotype 2 is the highest-risk stable state (31.7% mortality). Phenotype 1→0 is the most common transition, possibly reflecting stabilization/improvement.

## E014 — S2-light stride=12h sensitivity analysis
- **Date:** 2026-03-19
- **Method:** Same as E011-E013 but stride=12h (3 windows: [0,24), [12,36), [24,48)). Label permutation matching via Hungarian algorithm on centroid cosine distance.
- **Data:** S0 processed, 11,986 patients, frozen S1.5 encoder
- **Results:**
  - Centroid matching cosine distance: 0.0000–0.0001 (near-perfect alignment)
  - Stable fraction: 65.6% (vs 64.8% at stride=6h)
  - Non-self transition proportion: 19.1% (vs 10.4% — higher with less overlap)
  - Mortality ordering: [0,3,1,2] — identical to stride=6h
  - Highest-risk phenotype: P2 at 31.9% (vs 31.7%)
  - Mortality range: 28.0pp (vs 27.7pp)
  - Robustness: 3/4 criteria passed (MOSTLY ROBUST)
- **Artifacts:** data/s2/sensitivity_stride12/, data/s2/sensitivity_comparison.json
- **Conclusion:** Core mortality findings fully robust. Transition rates are overlap-sensitive but in the conservative direction — stride=6h underestimates transitions, not overestimates them.

## E015 — S3 cross-center temporal phenotype validation
- **Date:** 2026-03-19
- **Method:** Applied S2-light temporal phenotyping (KMeans trained on Center A windows) to Center B. Compared 6 criteria: stable fraction, transition proportion, mortality ordering, highest-risk phenotype, mortality range, prevalence L1.
- **Data:** Center A: 7,989 patients (train+val), Center B: 3,997 patients (test). Same frozen S1.5 encoder, same KMeans model.
- **Results:**
  - All 6/6 criteria passed: STRONG CROSS-CENTER GENERALIZATION
  - Stable fraction: A=65.0%, B=64.4% (diff=0.6pp)
  - Non-self transition proportion: A=10.3%, B=10.6% (diff=0.3pp)
  - Mortality ordering: identical [P0, P3, P1, P2]
  - Highest-risk: P2 in both (A=32.6%, B=30.0%)
  - Center B mortality range: 25.8pp
  - Per-window silhouette: within 0.001 across centers at all 5 positions
- **Artifacts:** data/s3/cross_center_report.json
- **Conclusion:** Temporal phenotype trajectories generalize from Center A to the independent Center B hospital. The encoder trained on Center A produces clinically consistent phenotypes on unseen data. This is the strongest cross-center evidence in the project.

## E016 — Frozen S1.5 downstream mortality classifier
- **Date:** 2026-03-23
- **Method:** LogisticRegression on frozen `data/s15/embeddings_s15.npy`; train on train split, tune threshold on validation split by balanced accuracy, evaluate on held-out test split.
- **Data:** Same S0 train/val/test split used throughout the project (Center A train+val, Center B test).
- **Results:**
  - Threshold selected on validation balanced accuracy: 0.575
  - Val: accuracy=0.801, balanced_accuracy=0.747, F1=0.488, AUROC=0.826
  - Test: accuracy=0.794, balanced_accuracy=0.738, recall=0.681, AUROC=0.829
  - Majority-class baseline accuracy on test: 0.854
- **Artifacts:** data/s15/mortality_classifier_report.json
- **Conclusion:** The learned S1.5 embedding space supports downstream mortality discrimination, but raw accuracy alone is misleading because mortality prevalence is only about 14%.

## E017 — Fresh S1.5 retraining + downstream mortality validation
- **Date:** 2026-03-23
- **Method:** Re-trained S1.5 from scratch in an isolated output directory, extracted new embeddings, then trained the same frozen-embedding LogisticRegression classifier. Also ran a threshold sensitivity check optimizing for validation accuracy instead of balanced accuracy.
- **Data:** S0 processed tensors, same train/val/test split.
- **Results:**
  - Fresh S1.5 pretraining: best val_loss=0.230 (epoch 47), reproducing the original Stage 1.5 training regime
  - Balanced-threshold classifier (`thr=0.55`): test accuracy=0.784, balanced_accuracy=0.745, precision=0.372, recall=0.691, F1=0.484, AUROC=0.829
  - Accuracy-optimized threshold (`thr=0.85`): test accuracy=0.865, balanced_accuracy=0.623, recall=0.280, AUROC=0.829
  - Majority-class baseline accuracy on test: 0.854
- **Artifacts:** data/s15_trainval/pretrain_log.json, data/s15_trainval/mortality_classifier_report.json, data/s15_trainval_accuracy/mortality_classifier_report.json
- **Conclusion:** A fresh retrain reproduces the representation quality of S1.5 and confirms that the main added value of the supervised validation is discrimination under imbalance-aware metrics, not headline accuracy.

## E018 — Advanced downstream mortality classifier with multi-modal feature fusion
- **Date:** 2026-03-23
- **Method:** HistGradientBoosting on richer S0-derived features: 48h statistical summaries, observation-density features, proxy indicator summaries, and non-leaky static metadata (age, sex, height, weight, ICU type).
- **Data:** Same S0 train/val/test split; no extra patients added, but more of the existing cohort information is used than in the embedding-only linear probe.
- **Results:**
  - Val: accuracy=0.795, balanced_accuracy=0.795, recall=0.795, AUROC=0.865
  - Test: accuracy=0.791, balanced_accuracy=0.780, precision=0.391, recall=0.764, F1=0.517, AUROC=0.862
  - Improvement over E017 balanced-threshold logistic probe: +0.7pp accuracy, +3.5pp balanced accuracy, +3.3 AUROC points
- **Artifacts:** data/s15_trainval/advanced_hgb/advanced_mortality_classifier_report.json
- **Conclusion:** Using more available modalities and a nonlinear classifier improves both plain accuracy and imbalance-aware metrics over the embedding-only linear model.

## E019 — Advanced downstream HGB ensemble (fused + stats views)
- **Date:** 2026-03-23
- **Method:** Two-view ensemble of HistGradientBoosting models: one on `fused_all` (S1.5 embeddings + statistics + masks + proxy + static) and one on `stats_mask_proxy_static`, with ensemble weight and threshold selected on validation balanced accuracy.
- **Results:**
  - Selected weights: fused=0.75, stats=0.25, threshold=0.07
  - Val: accuracy=0.766, balanced_accuracy=0.801, recall=0.850, AUROC=0.871
  - Test: accuracy=0.765, balanced_accuracy=0.785, recall=0.812, F1=0.503, AUROC=0.865
- **Artifacts:** data/s15_trainval/advanced_hgb_ensemble/advanced_mortality_classifier_report.json
- **Conclusion:** The ensemble gives the strongest balanced accuracy and AUROC in the repo so far, at the cost of lower plain accuracy due to a recall-oriented operating point.

## E020 — PhysioNet 2019 auxiliary data bridge
- **Date:** 2026-03-23
- **Method:** Converted local PhysioNet/CinC 2019 Sepsis Challenge stubs into the project's S0-compatible layout (`continuous`, `masks`, `proxy`, `static`, `splits`) using a shared 48h window. Reused PhysioNet 2012 preprocessing statistics to keep the auxiliary source numerically aligned with the pretrained S1.5 encoder.
- **Data:** 40,331 ICU stays from `archive/sepsis2019_stubs`, random stratified split by `sepsis_label`.
- **Results:**
  - Shared feature coverage: 18 / 21 continuous channels (`gcs`, `sodium`, `pao2` unavailable)
  - Sepsis prevalence: 4.65%
  - Overall missing rate before imputation: 78.7%
  - Split sizes: train=28,231 / val=6,050 / test=6,050
- **Artifacts:** data/s19_bridge/bridge_report.json, data/s19_bridge/splits.json, data/s19_bridge/processed/preprocess_stats.json
- **Conclusion:** A real second ICU time-series source is now integrated into the same on-disk interface used by S0, enabling multi-source supervised adaptation without refactoring the core data loaders.

## E021 — End-to-end supervised fine-tuning with auxiliary Sepsis 2019 transfer
- **Date:** 2026-03-23
- **Method:** Initialized the S1.5 encoder from the pretrained checkpoint, performed auxiliary supervised training on the bridged PhysioNet 2019 sepsis task, reset the prediction head, then fine-tuned end-to-end on PhysioNet 2012 mortality using an attention-pooled classifier head.
- **Data:** Auxiliary stage on `data/s19_bridge` (`sepsis_label`), main stage on `data/s0` cross-center mortality split.
- **Results:**
  - Auxiliary sepsis task: test accuracy=0.838, balanced_accuracy=0.809, recall=0.776, AUROC=0.888
  - Main mortality task: test accuracy=0.795, balanced_accuracy=0.753, precision=0.388, recall=0.692, F1=0.498, AUROC=0.842
  - Improvement over the frozen balanced-threshold embedding probe (E017): +1.1pp accuracy, +0.8pp balanced accuracy, +1.3 AUROC points
- **Artifacts:** data/s15_trainval/finetune_supervised/finetune_report.json, data/s15_trainval/finetune_supervised/supervised_auxiliary_history.json, data/s15_trainval/finetune_supervised/supervised_main_history.json
- **Conclusion:** End-to-end adaptation plus auxiliary transfer improves on the frozen embedding probe and slightly improves plain accuracy over the best single HGB model, while preserving substantially higher recall than accuracy-only operating points.

## E022 — Accuracy-oriented systematic downstream hyperparameter search
- **Date:** 2026-03-23
- **Method:** Exhaustive 35-run search over LogisticRegression, HistGradientBoosting, and HGB ensemble variants on `embeddings`, `embeddings_static`, `stats_mask_proxy_static`, and `fused_all` views. Decision thresholds were selected on validation accuracy to explicitly study the accuracy frontier under class imbalance.
- **Data:** Same S0 cross-center mortality split used throughout the project.
- **Results:**
  - Validation-accuracy leader: `ensemble d=3 lr=0.05 iter=200`, test accuracy=0.871, balanced_accuracy=0.660, precision=0.601, recall=0.361, AUROC=0.863
  - Validation-AUROC leader under the same search: `ensemble d=5 lr=0.05 iter=200`, test accuracy=0.874, balanced_accuracy=0.685, precision=0.603, recall=0.417, AUROC=0.867
  - Both exceed the majority-class baseline accuracy of 0.854, but both do so with markedly lower recall than the balance-oriented ensemble in E019
- **Artifacts:** data/s15_trainval/hparam_search_advanced/search_report.json
- **Conclusion:** Systematic search can push headline accuracy substantially higher, but the best accuracy-oriented operating points suppress sensitivity; E019 remains the preferred configuration when balanced accuracy and recall matter more than raw accuracy.

## E023 — Leakage-aware OOF stacking committee with validation/explanation
- **Date:** 2026-03-24
- **Method:** Combined the original train and validation patients into a development split, generated 5-fold stratified out-of-fold predictions from three base learners (`stats_mask_proxy_static` HGB, `fused_all` HGB, `fused_all` LogisticRegression), then fit a logistic meta-classifier on the OOF predictions. Evaluated threshold-specific operating points plus bootstrap confidence intervals, calibration, and permutation importance over meta-features.
- **Data:** Same S0 cross-center mortality split; train+val used only through OOF predictions, test remains the held-out Center B split.
- **Results:**
  - Accuracy operating point: test accuracy=0.880, balanced_accuracy=0.653, precision=0.682, recall=0.333, F1=0.448, AUROC=0.873
  - Balanced operating point on the same probabilities: test accuracy=0.803, balanced_accuracy=0.792, precision=0.409, recall=0.776, F1=0.536, AUROC=0.873
  - F1 operating point: test accuracy=0.854, balanced_accuracy=0.762, precision=0.501, recall=0.631, F1=0.558, AUROC=0.873
  - Bootstrap test AUROC 95% CI: [0.858, 0.888]; bootstrap test accuracy 95% CI at the accuracy threshold: [0.870, 0.889]
  - Meta-feature permutation importance ranked `fused_lr` first (mean AUROC drop 0.067), followed by `fused_hgb_d5` (0.056) and `stats_hgb_d5` (0.014)
  - Calibration remained weak despite strong ranking: Brier=0.144, ECE=0.222
- **Artifacts:** data/s15_trainval/stacking_accuracy/stacking_mortality_classifier.pkl, data/s15_trainval/stacking_accuracy/stacking_mortality_classifier_report.json, data/s15_trainval/stacking_accuracy/stacking_validation_report.json
- **Conclusion:** Using OOF stacking lifts both the best held-out accuracy and the best held-out AUROC in the repository, but the most accurate operating point suppresses recall substantially and is not well calibrated.

## E024 — DuckDB MIMIC profile and analysis-table readiness audit
- **Date:** 2026-03-24
- **Method:** Profiled the local `archive/db/mimic4.db` DuckDB database for schema inventory, key table row counts, cohort summary, first-day missingness, and readiness of all tables required by `src/build_analysis_table.py`.
- **Data:** Local MIMIC-IV mock DuckDB database used for schema-compatibility and concept-pipeline smoke testing.
- **Results:**
  - Schemas present: `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`
  - `mimiciv_derived.icustay_detail`: 15 ICU stays / 15 patients
  - `mimiciv_derived.sepsis3`: 5 stays (33.3% prevalence in the mock cohort)
  - Required analysis tables present: 9 / 9
  - First-day vital/lab summary tables had 0.0 missingness for the profiled mock rows
- **Artifacts:** data/mimic_db_profile/mimic_duckdb_profile.json, data/mimic_db_profile/mimic_duckdb_profile.md
- **Conclusion:** The legacy DuckDB path is queryable and complete enough to support reproducible cohort inspection and analysis-table construction, even though the bundled database remains a small mock rather than a research-scale MIMIC extract.

## E025 — Demo-ready MIMIC/eICU integration smoke validation
- **Date:** 2026-03-24
- **Method:** Added formal demo ingestion entry points for both external ICU sources. `scripts/prepare_mimic_demo.py` was run end-to-end on the local `archive/mimic-iv-mock` raw CSVs, rebuilding DuckDB tables, executing all MIMIC concepts SQL, and exporting patient-level analysis tables. `tests/test_eicu_loader.py` exercised the new raw-table eICU loader against a synthetic fixture shaped like the official demo tables, including vitals, labs, vasopressors, ventilation, and dialysis markers.
- **Data:** Local MIMIC-IV mock raw CSVs for the MIMIC smoke test; synthetic two-patient eICU fixture for unit validation. Official PhysioNet demo files were not available from the current environment.
- **Results:**
  - MIMIC smoke path completed successfully: `31` raw tables imported, `63 / 63` concept SQL files executed, `15` ICU stays exported to `720` hourly time-series rows
  - eICU loader built a valid `(2, 48, 17)` tensor and correctly recovered mortality, shock proxy, vasopressor exposure, ventilation intervals, dialysis proxy, and PaO2/FiO2 fallback
  - New reproducible entry scripts added: `scripts/prepare_mimic_demo.py`, `scripts/prepare_eicu_demo.py`
  - New raw-table module added: `src/eicu_loader.py`
- **Artifacts:** `/tmp/mimic_demo_out/mimic_demo_report.json` from the smoke run, plus local test-only eICU outputs created during `tests/test_eicu_loader.py`
- **Conclusion:** The repository is now formally wired for local MIMIC-IV-demo and eICU-CRD-demo files. The remaining blocker is data access, not missing integration code.

## E026 — Unified full-data MIMIC-IV / eICU transfer runs with isolated outputs
- **Date:** 2026-03-24
- **Method:** Ran the legacy V1 static clustering/evaluation pipeline through the unified `src/main.py` entry on full credentialed local MIMIC-IV 3.1 and eICU-CRD 2.0 extracts. The MIMIC run reused prepared analysis tables and wrote tagged outputs via `--tag mimic_real`. The eICU run reused the cached tensor in `data/processed_eicu_real` and wrote to explicit isolated output directories via `--output-reports-dir outputs/reports/eicu_real --output-figures-dir outputs/figures/eicu_real`.
- **Data:** Full local MIMIC-IV extract (`94,458` ICU stays; `41,295` Sepsis-3 stays) and full local eICU extract (`200,859` ICU stays).
- **Results:**
  - MIMIC-IV: silhouette=`0.1035`, CH=`7971.51`, DB=`2.7268`, valid survival rows=`94,435`
  - MIMIC-IV risk structure: dominant cluster `51.0%` of stays with `27.7%` mortality; smaller shock-enriched cluster `7.0%` of stays with `79.8%` mortality and `77.4%` shock
  - eICU: silhouette=`0.2135`, CH=`14461.68`, DB=`2.3771`, valid survival rows=`199,646`
  - eICU risk structure: dominant cluster `67.8%` of stays with `4.3%` mortality; shock-enriched cluster `22.5%` of stays with `21.1%` mortality and `46.2%` shock
  - Output isolation confirmed: MIMIC and eICU reports were written to separate directories without filename collisions
- **Artifacts:** `db/mimic4_real.db`, `data/processed_mimic_real/`, `data/processed_eicu_real/`, `outputs/reports/mimic_real/evaluation_report_mimic_real.json`, `outputs/reports/eicu_real/evaluation_report.json`
- **Conclusion:** The unified entry can now execute real full-database MIMIC-IV and eICU transfer runs end-to-end. These results broaden external evidence for ingestion and static risk structure, but they are not yet a full external replication of the Stage 3 temporal trajectory workflow.

## E027 — External temporal S1.5 + Stage 3 smoke validation on real MIMIC-IV / eICU subsets
- **Date:** 2026-03-24
- **Method:** Implemented `scripts/run_external_temporal_stage3.py`, which prepares source-specific external S0 bundles with `s0/external_temporal_builder.py`, reuses `data/s0/processed/preprocess_stats.json`, extracts frozen S1.5 embeddings from `data/s15/checkpoints/pretrain_best.pt`, then runs rolling-window Stage 3 clustering, transition analysis, and figures. Executed real-data smoke runs with `--max-patients 256` for both sources.
- **Data:** Prepared full local MIMIC-IV analysis tables under `data/processed_mimic_real/` and cached full local eICU tensor under `data/processed_eicu_real/`, subsetted to the first `256` stays per source for engineering validation.
- **Results:**
  - MIMIC-IV subset: pre-imputation missing rate `56.5%`, overall window silhouette `0.123`, stable fraction `50.0%`, non-self transitions `13.8%`
  - eICU subset: pre-imputation missing rate `84.5%`, overall window silhouette `0.249`, stable fraction `36.7%`, non-self transitions `18.4%`
  - Both runs produced the full external artifact tree: `s0/`, `s15/embeddings_s15.npy`, `s2/rolling_embeddings.npy`, `s2/trajectory_stats.json`, and Stage 3 figures
- **Artifacts:** `data/external_temporal_smoke/mimic/`, `data/external_temporal_smoke/eicu/`, `data/external_temporal_smoke/external_temporal_runs.json`
- **Conclusion:** The external temporal transfer path now works end-to-end on both databases. The remaining gap is full-cohort runtime, not missing integration code.

## E028 — Full-cohort external temporal S1.5 + Stage 3 replication on local MIMIC-IV / eICU
- **Date:** 2026-03-24
- **Method:** Executed `scripts/run_external_temporal_stage3.py` on the full prepared MIMIC-IV and eICU cohorts, reusing the PhysioNet 2012 preprocessing statistics and frozen S1.5 checkpoint for both sources. During the full run we fixed `--device auto` handling so PyTorch receives a valid `map_location`, and updated the runner so per-source summaries merge into `data/external_temporal/external_temporal_runs.json` instead of overwriting each other.
- **Data:** Full local MIMIC-IV 3.1 prepared tables (`94,458` ICU stays) and full local eICU-CRD 2.0 cached tensor (`200,859` ICU stays), aligned into `data/external_temporal/<source>/s0`.
- **Results:**
  - MIMIC-IV: mapped `15 / 21` channels, pre-imputation missing rate `55.5%`, overall window silhouette `0.1192`, stable fraction `50.9%`, single-transition fraction `41.3%`, multi-transition fraction `7.7%`, non-self transitions `14.3%`, stable-phenotype mortality range `17.1 pp`, top non-self transition `0 -> 3` with `16,718` events
  - eICU: mapped `12 / 21` channels, pre-imputation missing rate `81.4%`, overall window silhouette `0.1925`, stable fraction `43.1%`, single-transition fraction `45.8%`, multi-transition fraction `11.2%`, non-self transitions `17.3%`, stable-phenotype mortality range `12.2 pp`, top non-self transition `3 -> 1` with `26,561` events
  - Both full runs produced the complete artifact tree: `s0/`, `s15/embeddings_s15.npy`, `s2/rolling_embeddings.npy`, `s2/window_labels.npy`, `s2/trajectory_stats.json`, `s2/sanity_checks.json`, `s2/figures/`, per-source `run_summary.json`, and merged `data/external_temporal/external_temporal_runs.json`
- **Artifacts:** `data/external_temporal/mimic/`, `data/external_temporal/eicu/`, `data/external_temporal/external_temporal_runs.json`
- **Conclusion:** The repository now contains a full-cohort external temporal reproduction for both credentialed databases. The evidence is supplementary frozen-transfer validation under partial feature overlap, not source-specific retraining.
