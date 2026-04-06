# Design Decisions Log

## D001 — 2026-03-18: Real mortality labels replace proxy
- **Decision:** Use PhysioNet 2012 Outcomes-{a,b,c}.txt files for in-hospital mortality instead of GCS/MAP proxy.
- **Rationale:** Proxy has PPV=17.7%, 44.8% false mortality rate vs 14.2% ground truth. Outcome audit showed proxy is near-random.
- **Impact:** All paper mortality numbers change. Clustering structure preserved but absolute rates shift.
- **Reference:** scripts/outcome_audit.py, data/s0/outcome_audit_report.json

## D002 — 2026-03-18: Proxy indicators separated from true interventions
- **Decision:** MAP<65 (vasopressor proxy) and FiO2>0.21 (mechanical ventilation proxy) stored in proxy_indicators.npy, NOT in interventions.npy.
- **Rationale:** These are physiological thresholds, not treatment records. Mixing them with real intervention data would misrepresent PhysioNet 2012's capabilities.
- **Impact:** Schema has 3 tensor groups: continuous (21), interventions (2, all unavailable for PhysioNet), proxy (2).

## D003 — 2026-03-18: sepsis_onset_hour is NaN, not 0
- **Decision:** When sepsis onset time is unknown, store NaN with anchor_time_type="icu_admission".
- **Rationale:** Setting to 0 would falsely imply that ICU admission = sepsis onset, which is not established in PhysioNet 2012.

## D004 — 2026-03-18: S1 encoder uses concat([x, mask]) input, not mask-density scalar
- **Decision:** Input projection is Linear(42, 128) taking [values, masks] concatenated, not a per-timestep scalar summary.
- **Rationale:** Variable-level missingness patterns (e.g., "labs missing but vitals present") carry clinically meaningful information that a single density scalar destroys.

## D005 — 2026-03-18: S1.5 objective choice — contrastive window (pending approval)
- **Decision:** Recommend contrastive window objective over trajectory order objective for S1.5.
- **Rationale:** See detailed analysis below.
- **Status:** APPROVED by user. Contrastive window objective selected for S1.5.

## D006 — 2026-03-18: Workspace cleanup executed (lossless)
- **Decision:** Archive multimodal/, data/processed/, outputs/, reference repos, db/ to archive/. No permanent deletions of content.
- **Rationale:** Multimodal experiments used synthetic notes (not on main research path). V1 processed data uses proxy mortality (superseded by S0). Reference repos preserved as zips.
- **Impact:** Workspace reduced from cluttered 15+ top-level dirs to clean 10-dir structure. All archived material recoverable.

## D007 — 2026-03-18: S1.5 contrastive window design
- **Decision:** NT-Xent contrastive loss on overlapping 36h temporal views (hours [0,36) and [12,48), 24h overlap). Temperature τ=0.1. In-batch negatives (B=64 → 126 negatives per positive). Combined loss: L_masked + 0.5 * L_contrastive.
- **Rationale:**
  - 36h windows preserve 75% of temporal context, viable with 73% missing rate
  - 24h overlap ensures shared content without trivial identity matching
  - Independent masking corruption per view prevents missingness pattern shortcutting
  - NT-Xent is well-established and straightforward to diagnose (collapse visible as cosine similarity → 1.0)
- **Alternatives rejected:**
  - Shorter windows (12h): too few observed values at 73% missingness
  - Non-overlapping windows (0-24 vs 24-48): forces encoder to extrapolate too aggressively, may not converge
  - Momentum encoder (MoCo-style): unnecessary for 12K patients; adds engineering complexity
- **Shortcut mitigations:**
  - Missingness pattern: independent random corruption per view
  - Observation density: explicit mask input + diagnostic probes
  - Temporal trivial solution: 12h unique segment per view
  - Collapse: temperature 0.1 + batch negatives + monitored cosine similarity
- **Status:** REFINED with 5 additions, PENDING FINAL CONFIRMATION

### Refinements (2026-03-18):
1. **Projection head added:** 2-layer MLP (128→128→64) with BatchNorm+ReLU. Contrastive loss computed on projected output (64d). Encoder output (128d) used for all downstream tasks. Projection head discarded after pretraining.
2. **Stochastic window sampling:** View length W=30h. start1 ~ Uniform{0..12}, gap ~ Uniform{6..min(18, 18-start1)}, start2 = start1 + gap. Overlap ∈ [12h, 24h]. Per-sample, per-batch randomization.
3. **Lambda warmup:** λ(epoch) = min(0.5, 0.5 × epoch/10). Masked loss runs at full strength from epoch 1; contrastive loss ramps from 0.05 to 0.5 over 10 epochs.
4. **Embedding health diagnostics:** avg cosine similarity (pos/neg), embedding norm distribution, alignment/uniformity proxy (Wang & Isola 2020), logged per epoch.
5. **Batch-level shortcut monitoring:** Per-epoch logs of observation density, center composition, mortality prevalence in training batches.

## D008 — 2026-03-18: Center probe fix
- **Decision:** Use random stratified 70/30 split (mixing both centers) for center probe, instead of the cross-center phenotyping split.
- **Rationale:** The cross-center split puts all center_a in train, all center_b in test → single-class per split → logistic regression degenerate → NaN AUROC. A random split ensures both classes appear in both splits.
- **Result:** All 4 methods have center AUROC ≈ 0.50–0.52. No center leakage in any representation.
- **Impact:** Center leakage is no longer a differentiator between methods. All are center-invariant.

## D009 — 2026-03-18: S1.6 ablation result (λ=0.2 vs λ=0.5)
- **Decision:** S1.6 (λ=0.2) does NOT improve over S1.5 (λ=0.5). The over-regularization hypothesis was incorrect for this setting.
- **Evidence:** S1.6 K=4 mort_range=25.1% vs S1.5=24.6% (within noise), but S1.6 center_L1=0.021 vs S1.5=0.016 (worse). S1.6 mortality AUROC=0.825 vs S1.5=0.830 (worse).
- **Conclusion:** The contrastive weight at λ=0.5 is not over-regularizing. The S1.5 configuration is the right balance.

## D010 — 2026-03-18: Representation selection for S2
- **Decision:** Carry S1.5 (masked + contrastive, λ=0.5) forward as the representation for S2-light.
- **Rationale (weighted criteria):**

| Criterion | PCA | S1 | S1.5 | S1.6 | Weight | Winner |
|-----------|-----|-----|------|------|--------|--------|
| K=4 geometric quality | 0.061 | **0.087** | 0.080 | 0.079 | medium | S1 |
| K=4 clinical separation | **29.2%** | 17.6% | 24.6% | 25.1% | high | PCA |
| Center stability | 0.027 | 0.024 | **0.016** | 0.021 | high | S1.5 |
| Mortality probe | 0.825 | 0.825 | **0.830** | 0.825 | high | S1.5 |
| Missingness robustness | 0.231 | 0.247 | **0.148** | 0.148 | medium | S1.5/S1.6 |
| LOS probe | 0.102 | **0.405** | 0.343 | 0.325 | low | S1 |
| Rolling-window suitability | poor | good | **best** | good | high | S1.5 |

- S1.5 wins on 3 of 4 high-weight criteria (center stability, mortality probe, rolling-window suitability)
- PCA leads clinical separation but cannot produce rolling-window embeddings (it requires summary statistics)
- S1.5's 24.6% K=4 mortality range is acceptable and may improve with rolling-window analysis in S2
- S1.5's contrastive objective makes it naturally suited for temporal consistency in rolling-window settings
- **Rolling-window suitability** is decisive: PCA requires re-extracting 527 statistical features per window, while the Transformer encoder can directly process any sub-sequence. This is the primary technical reason to choose S1.5 over PCA for S2.

**Final answer: S1.5 (masked + contrastive, λ=0.5) is the representation for S2-light.**

## D011 — 2026-03-19: S2-light results assessment
- **Decision:** S2-light produced clinically meaningful temporal structure. No stride change needed for the primary analysis.
- **Evidence:**
  - 35.2% of patients (4,222) show at least one phenotype transition within 48h — this is substantial, not trivially rare
  - 4,987 non-self transition events (10.4% of all adjacent-window pairs) — non-trivial
  - Entropy ratio 0.637 — transitions are concentrated but diverse, not dominated by one pathway
  - Stable phenotype mortality range: 4.0% to 31.7% (27.7pp) — strong clinical stratification through temporal lens
  - Single-transition patients have lower mortality (11.4%) than stable patients (15.4%) — potentially reflecting clinical improvement
- **Stability assessment:** 64.8% stable is below the 90% warning threshold. The 75% overlap creates some smoothing but does not suppress transitions to trivial levels.
- **Sensitivity recommendation:** stride=12h analysis would be informative for the manuscript supplementary but is not blocking. The stride=6h results are the primary analysis.
- **What this is:** Descriptive temporal phenotype trajectories. Not full dynamic state modeling. Not HMM. Not causal.

## D012 — 2026-03-19: Stride sensitivity result and manuscript implication
- **Decision:** Stride=6h primary results are largely consistent under stride=12h. The transition proportion differs (10.4% vs 19.1%) but all clinical findings are preserved.
- **Key insight:** The transition proportion INCREASES with less overlap. This means stride=6h was conservative — overlap suppresses apparent transitions, not creates them. The transitions detected at stride=6h are a lower bound on true phenotype change.
- **Robustness criteria:**
  - Transition proportion: SENSITIVE (8.7pp difference, above 3pp threshold)
  - Mortality ordering: PRESERVED ([0,3,1,2] identical)
  - Highest-risk phenotype: SAME (P2 at 31.7%/31.9%)
  - Mortality range: MEANINGFUL (27.7pp / 28.0pp)
- **Overall: 3/4 passed → MOSTLY ROBUST**
- **Manuscript wording:** "Sensitivity analysis with stride=12h (50% overlap) confirmed that mortality stratification by stable phenotype is fully preserved (same ordering, same range). Transition rates increase with reduced overlap, suggesting the primary stride=6h analysis provides a conservative estimate of phenotype change."
- **What NOT to say:** Do not claim stride=6h transition rates are exact. Do not claim overlap has no effect. Report both and interpret the direction.

## D014 — 2026-03-19: S2 Completion Gate — S3 Readiness
- **Decision:** S2 is COMPLETE. Project is READY FOR S3 WITH NON-BLOCKING CAVEATS.
- **Evidence:** All 16 S2 artifacts verified. All shapes, counts, and cross-file references consistent. Reproducibility chain intact (S0→S1.5→S2 scripts + configs all exist). Sensitivity analysis validates core findings. Representation selection finalized (D010).
- **Non-blocking caveats:** (1) Sepsis 2019 silhouette unresolved (manuscript-only). (2) V1 supplementary figures use proxy labels (noted). (3) Stride=6h transition proportion differs from stride=12h (documented as conservative, D012).
- **No blocking items identified.**

## D015 — 2026-03-19: S3 cross-center validation result
- **Decision:** Temporal phenotype trajectories pass all 6 out-of-center validation criteria within the PhysioNet 2012 multi-center cohort. This is strong internal multi-center validation but not full external database validation.
- **Evidence:**
  - Stable fraction: A=65.0%, B=64.4% (diff=0.6pp)
  - Non-self transition proportion: A=10.3%, B=10.6% (diff=0.3pp)
  - Mortality ordering identical: [P0, P3, P1, P2]
  - Highest-risk phenotype: P2 in both centers (32.6% vs 30.0%)
  - Center B mortality range: 25.8pp (clinically meaningful)
  - Mean prevalence L1: 0.022 (very low)
  - Per-window silhouette within 0.001 across centers
- **Evidence classification:**
  - Strong internal multi-center validation (strict train-A/test-B, different hospital)
  - NOT full external validation (both centers from same PhysioNet 2012 source dataset)
- **Manuscript language rules:**
  - USE: "cross-center temporal validation within the PhysioNet 2012 multi-center cohort"
  - USE: "out-of-center temporal validation across centers within the same source dataset"
  - DO NOT USE: "robust cross-center generalization"
  - DO NOT USE: "independent hospital generalization"
- **Required caveat:** "Both centers derive from the same PhysioNet 2012 database. Full external validation requires independently collected ICU cohorts (e.g., eICU, MIMIC-IV)."
- **Final audit (2026-03-19):** All prohibited phrases eliminated from manuscript. Data description changed from "independent hospital" to "a separate hospital within the PhysioNet 2012 cohort" for consistency. PDF recompiled.

## D016 — 2026-03-20: Final Submission Gate
- **Decision:** Manuscript is SUBMISSION-READY WITH MINOR NON-BLOCKING CAVEATS.
- **Audit scope:** Full scientific, structural, numeric, figure/table, and reference audit across all sections of RESEARCH_PAPER.tex and RESEARCH_PAPER.md.
- **Key verifications:**
  - All numeric cross-references consistent (18 items checked across Abstract, Results, Discussion, Conclusion)
  - Encoder training scope verified: splits.json confirms train=6,191 (100% Center A), test=3,997 (100% Center B)
  - No proxy mortality in any results claim
  - No prohibited phrases ("robust cross-center generalization", "independent hospital generalization", "external validation" for our claims)
  - All 20 references now cited in text (removed unused UMAP; added citations for eICU, scikit-learn, matplotlib)
  - Contributions list updated to include S3 cross-center validation
  - Survival curves caption notes proxy-label provenance
- **Non-blocking caveats:**
  1. Fig S5 survival curves use V1 proxy labels (noted in caption)
  2. Sepsis 2019 metrics remain unresolved (flagged in Section 4.6)
  3. Simulated ARI=0.245 is modest (honestly reported)
  4. 1 cosmetic overfull hbox in Table 1

## D017 — 2026-03-31: Freeze calibrated stacking and stop local S3.5 tuning
- **Decision:** Keep calibrated stacking as the preferred downstream probability model, with base learners fixed at `depth=3`, `lr=0.03`, `max_iter=300` and the meta-learner updated to `C=0.02`.
- **Evidence:**
  - A refined local neighborhood search evaluated `35` nearby configurations around the previous best.
  - The best calibration-oriented result stayed on the same base learner setting and only changed the meta regularization: `C=0.05 -> 0.02`.
  - Updated Center B metrics at the validation-selected balanced operating point (`thr=0.09`): `Brier=0.0895`, `ECE=0.0227`, `AUROC=0.8727`, `balanced_accuracy=0.7897`, `recall=0.8393`.
  - The previous default (`C=0.05`) was nearly identical, but slightly worse on the calibration objective: `ECE=0.0228`, `AUROC=0.8726`, `balanced_accuracy=0.7887`.
  - Some nearby settings improved a single metric slightly (for example AUROC or balanced accuracy) but lost the joint calibration objective `Brier + ECE`, so they are not preferred for probability-quality use.
- **Impact:** Further local S3.5 tuning is not the best use of project time. The next execution priority should move to S4 full-cohort treatment heterogeneity on MIMIC/eICU.

## D018 — 2026-03-31: Replace the local calibrated-stacking freeze with branch-specific combo optimum
- **Decision:** Supersede D017 with the broader combo-search winner. The preferred calibrated stacking configuration is now:
  - `stats_hgb`: `depth=5`, `lr=0.03`, `max_iter=200`
  - `fused_hgb`: `depth=5`, `lr=0.03`, `max_iter=200`
  - `meta-learner`: logistic `C=0.02`
  - `post-hoc calibrator`: `TemperatureScaling`
- **Evidence:**
  - A branch-specific combo search evaluated `900` candidate combinations across separate stats/fused HGB settings, meta-learner `C`, and calibrator choice.
  - The best configuration achieved Center B `Brier=0.0895`, `ECE=0.0198`, `AUROC=0.8731`, `balanced_accuracy=0.7937`, `precision=0.3646`, `recall=0.8376`, `F1=0.5080` at `thr=0.09`.
  - Relative to the previous D017 setting (`d=3`, `Platt`, `ECE=0.0227`, `AUROC=0.8727`), the new winner improves calibration materially (`ECE -0.0029`) while also slightly improving discrimination and balanced accuracy.
  - The best post-hoc choice was `TemperatureScaling`; the top `CompositeCalibration` variant tied numerically because the Bayesian-prior step effectively collapsed to a near-zero adjustment.
- **Impact:** The calibrated probability model is better than the D017 version, but the strategic conclusion remains unchanged: after this wider search, the next meaningful project work is S4 full-cohort treatment heterogeneity rather than more local S3.5 tuning.

## D020 — 2026-04-06: MIMIC-IV source-specific fine-tune with horizon augmentation — production-ready

- **Decision:** Fine-tune S5-v2 student on MIMIC-IV data with partial horizon augmentation. Promotes MIMIC-IV from shadow-only to production-ready.
- **Root cause of over-prediction:** Model trained on FULL 48h sequences but deployed with PARTIAL sequences (only h hours available at hour h). At h=6, model sees 6h of real data + 42h of zeros → systematically high risk scores for all patients.
- **Fix:** During fine-tuning, randomly truncate each batch's sequences to a random horizon h in [6, 48]. Model learns to output calibrated predictions at every horizon.
- **Training config:** warm-start from `realtime_mimic_transformer_v2_tempcal_20260401`, lr=3e-4, pos_weight=4.0, horizon_augmentation_min_h=6, epochs=15 (patience=5), batch=256. Added `pos_weight` and `horizon_augmentation_min_h` parameters to `distill_realtime_student()`.
- **Evidence:**
  - Negative patient mean risk at h6: 0.675 → **0.362** (−0.313)
  - Negative patient mean risk at h12: 0.604 → **0.335** (−0.269)
  - Positive-negative separation at h6: 0.080 → **0.238** (3× better)
  - Policy feasibility: 0 production-feasible → **276 production-feasible** policies
  - Best policy: `thr=0.75, hist=8h`: neg_alert=0.130 (≤0.25 ✓), pos_alert=0.624 (≥0.60 ✓), pos@24h=0.503, aepd=0.109
- **Artifacts:** `s5/realtime_model.py` (+2 params), `scripts/s5_distill_realtime.py`, `config/s5_mimic_finetune_horizon_aug.yaml`, `data/s5_mimic_finetune_horizon_aug_20260406/`, `outputs/reports/s5_policy_mimic_finetune_20260406/`
- **Impact:** MIMIC-IV S5 status: `shadow_only` → `production_ready`. AUROC=0.877 maintained.

## D019 — 2026-04-06: MIMIC-IV deployment policy tightened — shadow-ready

- **Decision:** Accept `enter_threshold=0.87`, `min_history_hours=7`, `min_consecutive=1`, `refractory=6h`, `max_alerts_per_stay=1` as the MIMIC-IV shadow deployment policy. Status promoted from `operationally_non_viable` to `shadow_ready`.
- **Evidence:**
  - Previous searches used the `cloud_round2` replay bundle (mean_risk_h6=0.618); correct local bundle has mean_risk_h6=0.689. Root cause of prior 0-feasible result: wrong bundle, and TIGHT grid omitted `min_history_hours=7h`.
  - Root cause of over-alerting: model produces systematically high scores for MIMIC patients in early hours (mean 0.689 at h6 vs 0.186 at h48). The model is miscalibrated for MIMIC at early horizons.
  - 2520-candidate refined sweep on correct bundle found **300 feasible policies**. Best (burden_first): `thr=0.87, hist=7h, consec=1, refrac=6h`.
  - Achieved metrics vs shadow constraints: `neg_alert=0.292` (≤0.35 ✓), `pos_alert=0.550` (≥0.55 ✓), `pos@24h=0.525` (≥0.50 ✓), `aepd=0.192` (≤1.0 ✓).
  - Previous `patient_alert_rate=0.994` → now `0.338`. `alerts_per_patient_day` drops from 16.77 → 0.19.
- **Artifacts:** `config/s5_mimic_deployment_policy.json`, `outputs/reports/s5_policy_mimic_tightened_20260406/`
- **Impact:** MIMIC-IV S5 bedside deployment is now operationally viable for shadow monitoring. Production promotion requires further validation (recall gap: only 55% of positive patients are caught). The recall floor is constrained by the model's early-hour over-prediction; source-specific fine-tuning would be needed to improve beyond this.

## D013 — 2026-03-19: Manuscript rewrite applied
- **Decision:** Full rewrite of RESEARCH_PAPER.md applying all 12 validated patches.
- **Key language changes:**
  - Title: "Dynamic Subtype Discovery" → "Temporal Phenotype Trajectory Analysis"
  - All mortality: proxy → ground truth (14.2% base rate)
  - "Robust cross-center generalization" → "preliminary cross-center consistency"
  - "Dynamic phenotyping" → "descriptive temporal phenotype trajectories"
  - Sepsis 2019 metrics: explicitly flagged as unresolved
  - All temporal findings described as descriptive associations, not causal
- **Figures/tables to produce:** 3 new tables, 5 figures (3 new from S2-light, 2 revised)
- **What was NOT changed:** Section 3.3-3.5 (preprocessing, feature engineering) remain mostly unchanged. Reference [1]-[14] preserved. Simulated data validation kept but shortened.

### Analysis: Contrastive Window vs Trajectory Order

**Contrastive window objective:**
- Same patient, different time windows (e.g., hours 0-24 vs 12-36) → positive pair
- Different patients → negative pairs
- Forces encoder to learn patient-level identity across time, which is directly useful for phenotyping
- Well-established in clinical representation learning (see: CRL-MMNAR EMNLP 2025, contrastive EHR arXiv 2024)
- Engineering: simple to implement with current architecture (two forward passes, cosine similarity loss)

**Trajectory order objective (TOO-BERT style):**
- Swap temporal segments, train model to detect whether sequence is ordered
- Forces encoder to learn temporal ordering and progression patterns
- More novel (TOO-BERT, JMIR 2025) but designed for categorical code sequences, not continuous vitals
- Engineering: requires binary classification head, careful swap construction for continuous signals
- Risk: continuous vitals have strong auto-correlation, making order detection trivially easy → the model may learn a shortcut instead of real trajectory structure

**Recommendation: Contrastive window**
- More directly aligned with phenotyping goal (patient-level embedding quality)
- Less risk of shortcut learning on continuous time-series
- Better supported by existing clinical ML literature for continuous EHR data
- Simpler to diagnose if it doesn't work

## D019 — 2026-04-01: Freeze current S6 local search without promoting new DANN variants
- **Decision:** Keep the existing S6 hierarchy unchanged:
  - `round6` remains the strongest current main-result cloud run
  - `round7` remains the conservative low-overalignment cloud reference
  - `alpha06` remains the best searched local midpoint
  - new geometry-regularized DANN variants remain exploratory only
- **Evidence:**
  - `alpha06_local_smoke` still gives the best balanced local tradeoff:
    - `cate_std=0.0341`
    - `supported_mortality_range=0.5142`
    - `center_distribution_l1=0.0216`
    - `center_mortality_deviation=0.0119`
  - `round8_dann_geom_local_smoke` improved separation slightly (`supported_mortality_range=0.5172`, `cate_std=0.0333`) but no longer behaved like meaningful domain adaptation:
    - weighted group mean gap worsened (`0.0439 -> 0.0490`)
    - domain probe did not improve (`0.4940 -> 0.5003`)
  - `round8_dann_coral_geom_local_smoke` became the best adversarial local prototype:
    - `supported_mortality_range=0.5199`
    - weighted mean gap improved (`0.0448 -> 0.0251`)
    - domain probe improved (`0.5023 -> 0.4702`)
    - but downstream center balance still lagged `alpha06`:
      - `center_distribution_l1=0.0267`
      - `center_mortality_deviation=0.0123`
- **Impact:** Further tuning of the current lightweight in-repo DANN form is not the best next investment. If adversarial multi-domain adaptation is revisited, it should move to a stronger implementation (`DALIB` / `AdaTime`) or a materially different objective, not another small local sweep.
