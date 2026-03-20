# MedPatch: Confidence-Guided Multi-Stage Fusion for Multimodal Clinical Data

**Source**: arXiv 2508.09182 (2025)
**URL**: https://arxiv.org/html/2508.09182
**Relevance**: First integration of 4 clinical modalities (EHR + CXR + radiology reports + discharge notes)

---

## Overview

MedPatch integrates four routinely-collected clinical data modalities for ICU prediction tasks through a novel confidence-guided multi-stage fusion approach.

## Four Modalities
1. **Clinical time-series** (EHR from MIMIC-IV)
2. **Chest X-Ray images** (from MIMIC-CXR)
3. **Radiology Reports** (from MIMIC-Notes)
4. **Discharge Notes** (from MIMIC-Notes)

## Architecture Components

### Unimodal Encoders
- LSTM for structured EHR data
- Vision Transformer for CXR images
- BioBERT for textual reports

### Confidence Predictors
Token-level confidence scores via calibrated temperature scaling. Applied directly to raw logits to align confidence across heterogeneous modalities.

### Confidence-Based Patching
- High/low-confidence token clusters (0.75 threshold)
- Pooled separately through distinct classifiers
- Creates parallel prediction pathways

### Missingness Module
Explicit indicator vector tracking modality availability, enabling processing of incomplete patient records.

### Late Fusion Layer
Adaptive softmax-normalized weights combine predictions from all components.

## Results
- **Mortality prediction** (3 modalities): AUROC 0.876 (vs 0.872 ensemble baseline)
- **Clinical condition classification** (4 modalities): AUROC 0.862, AUPRC 0.614
- **Discharge notes alone**: AUROC 0.856 for mortality (strong unimodal signal)
- **40-90x fewer trainable parameters** than competing transformer approaches

## Key Takeaways for Our Project
- Discharge notes contain very strong predictive signal for mortality
- Confidence-guided fusion is more effective than uniform weighting
- Missingness module is essential for handling incomplete data
- Multi-stage fusion (joint + late) outperforms single-strategy approaches
- Parameter efficiency matters for practical deployment
