# Sepsis Subtype Gold-Standard Prediction Spec

Date: 2026-04-03

This note defines the most defensible "gold-standard" targets for predicting
all major subtype systems already discussed in this project. The key rule is
that we should not collapse every subtype family into a single flat label,
because each family is grounded in a different biological or clinical
measurement space.

## 1. What Counts As Gold Standard

A subtype target is `gold-standard` only if one of the following is true:

1. The original paper published patient-level subtype assignments or a
   reproducible assignment rule.
2. The subtype is anchored to a validated biomarker threshold used for
   prospective enrichment or randomized treatment allocation.
3. The subtype score comes from a published molecular signature with external
   validation.

Everything else is a `proxy target`, which is still useful for model training
but should not be represented as if it were identical to the published subtype.

## 2. Unified Task Layout

We should model subtype prediction as a multi-task problem with both
classification and regression heads.

Classification heads
- `immune_state_label`
- `clinical_phenotype_label`
- `trajectory_phenotype_label`
- `fluid_strategy_group_label`

Regression heads
- `mals_score`
- `immunoparalysis_score`
- `alpha_beta_gamma_delta_logits`
- `trajectory_group_membership_logits`
- `restrictive_fluid_benefit_score`
- `balanced_crystalloid_benefit_score`

Clinical recommendation heads
- `immune_therapy_candidate_score`
- `fluid_strategy_recommendation_score`
- `vasopressor_support_intensity_score`

## 3. Gold-Standard Targets By Subtype Family

### A. Immune Endotypes

#### A1. MALS / MAS-like hyperinflammatory state

Gold-standard label source
- Kyriazopoulou et al. 2017.
- PROVIDE 2022.
- ImmunoSep 2026.

Gold-standard features
- Ferritin, especially `> 4420 ng/mL`.
- Platelet suppression / coagulopathy.
- Liver injury markers such as bilirubin, ALT, AST.
- Hyperinflammatory cytokines when available.

What is truly gold-standard here
- `Ferritin > 4420 ng/mL` is the cleanest operational anchor.
- Trial-grade labels are best represented as `MALS yes/no` plus a continuous
  hyperinflammation score.

Recommended modeling target
- Binary head: `MALS`.
- Continuous head: `mals_score = f(log ferritin, platelets, bilirubin, d_dimer,
  CRP, cytokines when available)`.

What the current repo should not do
- Do not claim that any bedside-only CRP/platelet rule is equivalent to the
  trial-grade MALS label.

#### A2. Immunoparalysis

Gold-standard label source
- PROVIDE 2022.
- ImmunoSep 2026.
- Monneret et al. 2025 real-world cohort on mHLA-DR.

Gold-standard features
- Monocyte HLA-DR measured by standardized flow cytometry.
- Persistent lymphopenia.
- Secondary infection burden.
- Blunted immune-response assays when available.

What is truly gold-standard here
- `mHLA-DR` is the anchor biomarker.
- Recent large cohort evidence supports low mHLA-DR as the most actionable
  enrichment biomarker of sepsis-induced immunosuppression.
- Exact thresholds are assay-specific, but recent literature supports a low
  range around `5000-8000 AB/C or molecules/monocyte` depending on protocol.

Recommended modeling target
- Binary head: `immunoparalysis`.
- Continuous head: `immunoparalysis_score`.
- Time-aware regression is essential because delayed persistent suppression is
  more meaningful than one single early measurement.

What the current repo should not do
- Do not call `EIL-like` a gold-standard immune endotype. That is a proxy label,
  not a published trial-grade immune class.

### B. Clinical Organ-Dysfunction Phenotypes

#### B1. Alpha / Beta / Gamma / Delta

Gold-standard label source
- Seymour et al. JAMA 2019.

Gold-standard features
- Original latent class assignments from the 29 clinical variables used in the
  paper.
- Organ dysfunction burden.
- Inflammation and shock markers.

What is truly gold-standard here
- The original `alpha/beta/gamma/delta` classes are latent phenotypes, not a
  hand-built ruleset.
- The gold-standard target is therefore the published class assignment or a
  reproducible reimplementation of the original latent class model.

Recommended modeling target
- 4-class softmax head: `clinical_phenotype_label`.
- Regression head: distance or probability to each phenotype centroid.
- If original class probabilities are unavailable, train on high-quality proxy
  labels but keep them explicitly marked as `proxy_alpha_beta_gamma_delta`.

What the current repo should not do
- Do not advertise deterministic SOFA thresholding as equivalent to Seymour's
  latent class definitions.

### C. Dynamic Vital-Sign Trajectory Phenotypes

#### C1. Group A / B / C / D

Gold-standard label source
- Bhavani et al. Intensive Care Medicine 2022.

Gold-standard features
- First `8 hours` of temperature trajectory.
- First `8 hours` of heart-rate trajectory.
- First `8 hours` of respiratory-rate trajectory.
- First `8 hours` of blood-pressure trajectory.

What is truly gold-standard here
- Posterior membership from the group-based trajectory model is the proper
  label, not a one-time static vital-sign threshold snapshot.

Recommended modeling target
- 4-class sequence head: `trajectory_phenotype_label`.
- Regression head: posterior probability for `A/B/C/D`.
- Treatment-response auxiliary head: balanced crystalloids benefit, especially
  for `Group D`.

This is the most deployable subtype family
- The inputs are universally available.
- The treatment interaction is among the cleanest replicated signals in the
  literature.
- It aligns naturally with the repo's stage-5 rolling monitor.

### D. Fluid-Strategy Benefit Subgroups

#### D1. Restrictive-fluid benefit subgroup

Gold-standard label source
- Zhang et al. Nature Communications 2024.

Gold-standard features
- Transcriptome-derived benefit score.
- Six-protein proteomic signature used to predict benefit score.
- Septic-shock context with vasopressor dependence and fluid-management data.

What is truly gold-standard here
- The most faithful target is the published `benefit score` in transcriptome
  space.
- The six-protein panel is a surrogate predictor of that benefit score, not the
  causal label itself.

Recommended modeling target
- Regression head: `restrictive_fluid_benefit_score`.
- Binary threshold head: `benefit_from_restrictive_strategy`.
- Proteomic distillation head when only proteins are available.

What the current repo should not do
- Do not reduce this to `high_benefit / low_benefit` based only on lactate and
  blood pressure if you want to call it gold-standard.

## 4. How To Predict All Subtypes In One Model

The architecture should be hierarchical, not flat.

Stage 1: universal patient state encoder
- Structured physiology sequence.
- Treatments and organ support.
- Optional notes or molecular features.

Stage 2: family-specific heads
- Immune head.
- Clinical phenotype head.
- Trajectory head.
- Fluid-benefit head.

Stage 3: recommendation layer
- Map subtype probabilities plus treatment-effect scores into ranked
  recommendations.
- Emit `recommendation`, `confidence`, `evidence_family`, and `not_for_autonomy`
  flags.

## 5. Loss Design

Recommended loss bundle
- Cross-entropy for categorical subtype heads.
- KL divergence when teacher or posterior distributions are available.
- MSE / Huber for continuous benefit scores.
- Pairwise ranking loss for treatment-benefit ordering.
- Masked losses because not every dataset has every subtype family.

This is the key implementation rule
- Different datasets supervise different heads.
- Do not throw away a patient just because one subtype family is missing.

## 6. Practical Label Strategy For This Repo

### Level 1: true or near-gold labels

- `MALS` when ferritin-based trial anchor is available.
- `immunoparalysis` when standardized mHLA-DR or validated equivalent is
  available.
- `trajectory A/B/C/D` when early-hour vitals are complete enough to reproduce
  the Bhavani trajectory family.
- `restrictive_fluid_benefit_score` when molecular data or validated teacher
  labels are available.

### Level 2: high-quality proxy labels

- `proxy_alpha_beta_gamma_delta` from bedside clinical data.
- `proxy_fluid_benefit` from shock plus cardio-renal and lung-risk features.
- `proxy_immune_state` from ferritin, CRP, platelets, bilirubin, lymphocytes,
  and infection persistence when mHLA-DR is absent.

### Level 3: distillation targets

- Teacher embeddings from causal phenotyping and phenotype naming modules.
- Posterior subtype probabilities from clustering or mixture models.
- Continuous causal-benefit estimates from stage 4 and stage 6 artifacts.

## 7. Immediate Build Plan

1. Rename current repo labels into explicit `proxy_*` names where needed.
2. Add a new multi-task label schema with masked supervision.
3. Make the stage-5 realtime path predict:
   - deterioration risk
   - trajectory A/B/C/D
   - immune-state probability
   - fluid-benefit score
4. Keep alpha/beta/gamma/delta as a separate clinical phenotype head, but do
   not present it as gold-standard unless original latent-class reproduction is
   implemented.

## 8. Source Links

- MALS / ferritin anchor:
  https://pubmed.ncbi.nlm.nih.gov/28918754/
- PROVIDE:
  https://pubmed.ncbi.nlm.nih.gov/36384100/
- ImmunoSep:
  https://pubmed.ncbi.nlm.nih.gov/41359996/
- mHLA-DR cohort:
  https://pubmed.ncbi.nlm.nih.gov/40986015/
- Seymour alpha/beta/gamma/delta:
  https://pubmed.ncbi.nlm.nih.gov/31104070/
- Bhavani trajectories:
  https://pubmed.ncbi.nlm.nih.gov/36152041/
- CMAISE fluid-strategy subgroups:
  https://pubmed.ncbi.nlm.nih.gov/39424794/
