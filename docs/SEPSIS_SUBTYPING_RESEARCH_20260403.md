# Sepsis Subtyping Research And Open Asset Map

Date: 2026-04-03

This note checks the subtype systems mentioned in the user brief against the
primary literature, separates validated findings from extrapolated rules, and
maps each line of work to data and tools that are either openly downloadable or
credential-restricted.

## 1. What Is Solidly Supported By Primary Papers

### A. Hyperinflammatory immune subtype: MALS / MAS-like

- Ferritin `> 4420 ng/mL` as a high-specificity marker for macrophage
  activation-like syndrome in sepsis comes from Kyriazopoulou et al. and was
  reused in later precision-immunotherapy trials.
- The 2022 PROVIDE trial stratified patients using ferritin `> 4420 ng/mL` for
  MALS and low monocyte HLA-DR for immunoparalysis, then tested targeted
  immunotherapy.
- The 2026 `ImmunoSep` randomized trial used the same immune-state logic and
  assigned IV anakinra to MALS and subcutaneous recombinant human interferon
  gamma to immunoparalysis. It improved organ dysfunction at day 9 but did not
  show a statistically significant 28-day mortality benefit.

Practical implication:
- If you want a model output that is closest to current interventional evidence,
  an immune-state head should distinguish at least `MALS`, `immunoparalysis`,
  and `intermediate`.

Important correction:
- Your brief attributes IFN-gamma as the core treatment for the low-HLA-DR
  subtype, which is supported by `ImmunoSep`, but the strongest older
  biomarker-guided randomized evidence in this space also includes `GM-CSF`.
  This means the therapy layer should not hard-code one agent as universally
  preferred without trial-context metadata.

### B. Clinical organ-dysfunction phenotypes: alpha / beta / gamma / delta

- The `alpha`, `beta`, `gamma`, and `delta` sepsis phenotypes were derived in
  Seymour et al. `JAMA 2019`.
- That paper showed differential baseline biology, outcomes, and potential
  treatment-effect heterogeneity, but these phenotypes were latent classes
  estimated from clinical variables, not a simple bedside thresholding rule.

Practical implication:
- The current repo's `src/subtype_label_engine.py` already uses a rule-based
  proxy version of organ-dominant subtype labels. Those are useful training
  proxies, but they are not identical to the original JAMA latent phenotypes.

### C. Dynamic vital-sign trajectories: Group A / B / C / D

- The four trajectory groups in the user brief match Bhavani et al.
  `Intensive Care Medicine 2022`.
- That study used vital signs from the first `8 hours` and found consistent
  subphenotypes with different outcomes.
- In a secondary SMART analysis, `Group D` had lower mortality with balanced
  crystalloids versus saline, which is one of the cleaner treatment-response
  signals among the papers you cited.

Practical implication:
- This is the strongest justification for adding a real-time trajectory subtype
  head to your bedside model, because the features are universally available and
  align with the repo's stage-5 streaming monitor.

### D. Fluid-response multi-omics subgroup

- The 2024 Nature Communications paper by Zhang et al. identified septic-shock
  subgroups with different fluid-strategy response and built a six-protein
  proteomic signature to predict benefit from restrictive fluid strategy.
- This is real and newer than most of the other phenotype literature.
- However, raw individual-level patient data are not fully open. Public pieces
  include source data, processed transcriptomic releases, and project
  documentation; raw transcriptomics require NGDC approval.

Practical implication:
- This line is best used to design a `fluid_strategy_benefit` head and a
  recommendation rule that emits confidence and evidence source, not as a hard
  guideline replacement.

## 2. What In The Brief Is More Aggressive Than The Evidence

- Several treatment bullets in the brief read like direct practice rules. The
  underlying papers mostly support `heterogeneity of treatment effect`,
  `candidate precision strategy`, or `trial-enrichment logic`, not universal
  one-to-one bedside directives.
- The alpha/beta/gamma/delta section is especially easy to overstate. Seymour
  2019 did not publish a simple deterministic bedside rubric equivalent to the
  rules in the brief.
- The fluid-strategy multi-omics paper supports subgroup-guided fluid strategy
  research, but not a ready-to-deploy standard-of-care protocol.

## 3. Openly Downloadable Assets

Downloaded by `scripts/download_sepsis_subtyping_assets.sh`:

Datasets
- `PhysioNet Challenge 2019 v1.0.0` training tree: open-access hourly sepsis
  onset data for early detection work. The script keeps this as an opt-in mirror
  because the site exposes thousands of small files rather than a stable bundle.
- `MIMIC-IV Demo v2.2`: openly downloadable demo subset useful for pipeline and
  SQL validation.
- `eICU-CRD Demo v2.0`: openly downloadable demo subset for external-schema
  validation.
- `GSE65682` series matrix: large public blood transcriptome cohort used in
  many sepsis endotype papers.
- `GSE185263` raw counts: newer public RNA-seq cohort focused on sepsis
  severity and endotypes.
- `OMIX006238` proteomics CSV: open protein-level dataset released from the
  CMAISE septic-shock fluid-strategy project.

Tools
- `MIT-LCP/mimic-code`: official community codebase for MIMIC concepts.
- `MIT-LCP/eicu-code`: official community codebase for eICU concepts.
- `microsoft/mimic_sepsis`: reproducible sepsis-cohort extraction code based on
  the AI Clinician lineage.
- `AI4Sepsis` and `CanIgetyoursignature` challenge source packages from the
  PhysioNet 2019 competition.

## 4. High-Value But Restricted Assets

- `MIMIC-IV v3.1`: credentialed PhysioNet access required.
- `eICU-CRD v2.0`: credentialed PhysioNet access required.
- `CMAISE` raw transcriptomics `PRJCA006118`: formal NGDC approval required.
- Full patient-level data from the 2024 fluid-strategy Nature paper are not
  publicly open due to privacy constraints.
- `OMIX006457` transcriptomic companion files in the same CMAISE project are
  listed as controlled-access on NGDC.

## 5. Recommended Model Targets For This Repo

Given the evidence and the current codebase, the most defensible target schema
is:

1. Mortality / deterioration risk
2. Immune subtype
   - `MALS`
   - `immunoparalysis`
   - `intermediate_or_unclassified`
3. Organ / clinical phenotype
   - proxy `alpha/beta/gamma/delta`
4. Dynamic trajectory phenotype
   - `A/B/C/D`
5. Treatment-benefit heads
   - `balanced_crystalloid_benefit`
   - `fluid_restriction_benefit`
   - `immune_therapy_candidate`

Each recommendation should emit:
- predicted subtype
- suggested therapy direction
- evidence family
- confidence
- warning that recommendation is research-supportive, not autonomous clinical
  decision-making

## 6. Primary Sources

- Kyriazopoulou E, et al. `Macrophage activation-like syndrome: an immunological
  entity associated with rapid progression to death in sepsis`. Crit Care. 2017.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/28918754/
- Giamarellos-Bourboulis EJ, et al. `Toward personalized immunotherapy in
  sepsis: The PROVIDE randomized clinical trial`. Nat Med. 2022.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/36384100/
- Giamarellos-Bourboulis EJ, et al. `Precision Immunotherapy to Improve Sepsis
  Outcomes: The ImmunoSep Randomized Clinical Trial`. JAMA. 2026.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/41359996/
- Seymour CW, et al. `Derivation, Validation, and Potential Treatment
  Implications of Novel Clinical Phenotypes for Sepsis`. JAMA. 2019.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/31104070/
- Bhavani SV, et al. `Development and validation of novel sepsis subphenotypes
  using trajectories of vital signs`. Intensive Care Med. 2022.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/36152041/
- Zhang Z, et al. `Identifying septic shock subgroups to tailor fluid strategies
  through multi-omics integration`. Nat Commun. 2024.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/39424794/

## 7. Data And Tool Sources

- PhysioNet Challenge 2019:
  https://physionet.org/content/challenge-2019/1.0.0/
- MIMIC-IV Demo v2.2:
  https://physionet.org/content/mimic-iv-demo/2.2/
- eICU Demo v2.0:
  https://physionet.org/content/eicu-crd-demo/2.0/
- MIMIC-IV v3.1:
  https://physionet.org/content/mimiciv/3.1/
- eICU-CRD v2.0:
  https://physionet.org/content/eicu-crd/2.0/
- GSE65682:
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65682
- GSE185263:
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185263
- CMAISE fluid-strategy paper:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11489719/
- OMIX006238:
  https://ngdc.cncb.ac.cn/omix/release/OMIX006238
- OMIX006457:
  https://ngdc.cncb.ac.cn/omix/release/OMIX006457
- MIT-LCP mimic-code:
  https://github.com/MIT-LCP/mimic-code
- MIT-LCP eicu-code:
  https://github.com/MIT-LCP/eicu-code
- microsoft mimic_sepsis:
  https://github.com/microsoft/mimic_sepsis
