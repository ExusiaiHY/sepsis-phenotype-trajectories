# Multimodal Graph Neural Networks in Healthcare: Review of Fusion Strategies

**Source**: Frontiers in Artificial Intelligence, January 2026
**URL**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1716706/full
**Authors**: Maria Vaida (Harrisburg University), Ziyuan Huang (UMass Chan Medical School)
**Relevance**: Comprehensive review of GNN-based multimodal fusion in clinical domains

---

## Overview

Systematic review of 85 studies (2020-2025) on how GNNs integrate diverse healthcare data sources for clinical decision-making.

## GNN Architectures Reviewed
- Graph Convolutional Networks (GCNs)
- Graph Attention Networks (GATs)
- GraphSAGE
- Heterogeneous GNNs (HGNNs)
- Relational Graph Convolutional Networks (RGCNs)

Enhanced with temporal encoders, VAEs, and GANs.

## Fusion Strategy Taxonomy
1. **Early Fusion**: Features concatenated before graph convolutions
2. **Intermediate Fusion**: Integration within graph layers using attention
3. **Late Fusion**: Independent modality processing with post-hoc alignment
4. **Hybrid Fusion**: Combinations of above strategies

## Clinical Application Domains
- **Pharmacology**: Drug-drug interaction, adverse event prediction (AUC 0.81-0.99)
- **Oncology**: Cancer detection, prognosis, histopathology + radiology + genomics fusion
- **Neuropsychiatry**: Mental health diagnosis, neurological condition prediction
- **Epidemiology**: Infectious disease forecasting
- **Clinical Operations**: Patient risk prediction, resource optimization
- **Genomics**: ncRNA-miRNA interaction, gene-protein-disease modeling

## Key Challenges
1. Data heterogeneity across modalities
2. Computational scaling for large clinical datasets
3. Interpretability and transparency

## Key Takeaways for Our Project
- GNNs are an advanced fusion option (not for MVP, but for future extensions)
- Attention-enhanced intermediate fusion is the current best practice for GNNs
- Patient similarity graphs are a natural way to model ICU cohorts
- Consider as a V3 extension after basic fusion methods are established
