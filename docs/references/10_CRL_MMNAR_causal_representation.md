# CRL-MMNAR: Causal Representation Learning from Multimodal Clinical Records

**Source**: EMNLP 2025 (Conference Paper #3359)
**URL**: https://aclanthology.org/2025.emnlp-main.1465.pdf
**Local PDF**: CRL_MMNAR_causal_representation_learning_EMNLP2025.pdf
**Relevance**: State-of-the-art missing modality handling with contrastive learning for ICU data

---

## Overview

CRL-MMNAR learns patient representations by fusing multimodal inputs with missingness embeddings and optimizing reconstruction and contrastive losses. A second stage predicts clinical outcomes with multitask heads and applies a rectifier to correct missingness-induced bias.

## Two-Stage Framework

### Stage 1: Representation Learning
- Missingness-aware transformation followed by attention-based fusion
- Two key objectives:
  1. Cross-modality reconstruction loss
  2. Contrastive alignment loss
- Encodes each patient's modality availability as a binary vector

### Stage 2: Outcome Prediction
- Multitask prediction heads for clinical outcomes
- Rectifier module to correct missingness-induced bias
- Handles non-random missingness patterns (sicker patients get more tests)

## Modalities Used
- Electronic Health Records (structured)
- Clinical notes (unstructured text)
- Chest radiographs (imaging)

## Evaluation
- Cohort: 20,000 adult ICU patients
- Demonstrates effectiveness on MIMIC data
- Handles realistic non-random missingness patterns

## Key Innovations
1. **Missingness embeddings** - Explicit encoding of which modalities are available
2. **Causal debiasing** - Corrects for non-random missingness (clinical data is not MCAR)
3. **Cross-modality reconstruction** - Learn to predict missing modality from available ones
4. **Contrastive alignment** - Align representations across modalities

## Key Takeaways for Our Project
- Non-random missingness is a real clinical problem (not Missing Completely At Random)
- Missingness embeddings are a simple but effective technique
- Cross-modality reconstruction as auxiliary loss improves representations
- Contrastive alignment provides modality-invariant features
- Two-stage approach (representation then prediction) is effective
