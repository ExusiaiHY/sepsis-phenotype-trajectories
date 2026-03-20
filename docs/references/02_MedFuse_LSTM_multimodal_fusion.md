# MedFuse: Multi-Modal Medical Data Fusion

**Source**: GitHub (nyuad-cai/MedFuse)
**URL**: https://github.com/nyuad-cai/MedFuse
**Relevance**: Reference implementation for EHR + CXR fusion with missing modality handling

---

## Overview

MedFuse is an LSTM-based fusion module that accommodates uni-modal and multi-modal input for clinical prediction. It introduces benchmark results for in-hospital mortality and phenotype classification using MIMIC-IV + MIMIC-CXR.

## Architecture

### Stage 1: Modality-Specific Pre-training
- Train imaging encoder on 14 radiology labels (MIMIC-CXR)
- Train temporal encoder on clinical time-series (MIMIC-IV)

### Stage 2: Fusion and Fine-tuning
- Integrate pre-trained representations using MedFuse module
- Fine-tune on target clinical tasks

## Supported Fusion Strategies
- `early` - Early feature concatenation
- `joint` - Joint learning
- `daft` - Dynamic Attention Fusion Transform
- `mmtm` - Multi-Modal Tensor Modulation
- `uni_ehr` - EHR-only unimodal baseline
- `uni_cxr` - CXR-only unimodal baseline
- `medfuse` - Proposed LSTM-based fusion (handles missing modalities)

## Clinical Tasks
1. In-hospital mortality prediction
2. Phenotype classification (25 labels)

## Key Innovation
MedFuse demonstrates robustness on partially paired test sets with missing CXR, outperforming complex fusion strategies when data is sparse. This is critical because in real ICU data, most patients lack paired imaging.

## Data Configuration
- Paired EHR + CXR samples
- EHR-only samples
- Mixed (partial) datasets
- All MIMIC-CXR radiology samples

## Key Takeaways for Our Project
- Pre-train encoders separately, then fuse and fine-tune
- The fusion module must handle missing modalities gracefully
- LSTM is sufficient for temporal EHR encoding as a baseline
- Compare at least 4 fusion strategies
