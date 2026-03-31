"""
s6_optimization - Model optimization addressing 8 key limitations.

Modules:
  - phenotype_naming:    Mechanism-based phenotype assignment using CATE + organ scores
  - missingness_encoder: Informative missingness modeling (SAITS-style attention masks)
  - causal_phenotyping:  End-to-end causal treatment effect pipeline for phenotype naming
  - baseline_comparison: Quantitative baseline-vs-S6 comparison reporting
  - run_comparison:      Quantitative comparison between two S6 run directories
  - severity_split_search: Search cluster-aware severity split targets on existing S6 runs
"""
