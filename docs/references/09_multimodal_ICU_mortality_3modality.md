# Multimodal ICU Mortality: Three-Modality Fusion Model

**Source**: GitHub (bbleier/multimodal-icu-mortality)
**URL**: https://github.com/bbleier/multimodal-icu-mortality
**Relevance**: Reference implementation for 3-modality (notes + CXR + labs) ICU mortality prediction

---

## Overview

Multi-modal fusion model for ICU mortality prediction in pleural effusion patients, integrating clinical notes, chest X-ray images, and laboratory results.

## Architecture

### Three Sub-Models
1. **Clinical Notes**: ClinicalBERT fine-tuned for text classification
2. **Chest X-Rays**: DenseNet121 (CheXnet) for image classification
3. **Lab Values**: Fully connected neural network

### Fusion Mechanism
Concatenation with adaptor layers. Flexible handling of 1, 2, or 3 simultaneous modalities.

## Performance
- **Fusion AUROC**: 0.82
- **Notes-only AUROC**: 0.75
- **Labs-only AUROC**: 0.74
- **CXR-only AUROC**: 0.67
- Consistent improvements across accuracy, precision, recall, and F1

## Data Sources
MIMIC-IV + MIMIC-CXR + MIMIC-IV-Note

## Key Takeaways for Our Project
- Notes are the strongest unimodal predictor (0.75 > 0.74 labs > 0.67 CXR)
- Simple concatenation fusion already achieves meaningful improvement over unimodal
- ClinicalBERT + DenseNet121 is a standard encoder combination
- Flexible modality handling (1/2/3 modalities) is important for clinical use
