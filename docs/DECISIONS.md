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
