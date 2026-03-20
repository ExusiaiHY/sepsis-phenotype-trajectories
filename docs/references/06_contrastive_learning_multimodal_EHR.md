# Contrastive Learning on Multimodal Analysis of Electronic Health Records

**Source**: arXiv 2403.14926 (March 2024, revised August 2025)
**URL**: https://arxiv.org/abs/2403.14926
**Authors**: Tianxi Cai, Feiqing Huang, Ryumei Nakada, Linjun Zhang, Doudou Zhou
**Relevance**: Theoretical foundation for multimodal contrastive learning on EHR

---

## Overview

Proposes a novel multimodal feature embedding generative model with multimodal contrastive loss for EHR representation learning. Bridges structured clinical codes and unstructured clinical notes.

## Key Contributions

### Methodological
- Novel multimodal contrastive loss function for joint EHR analysis
- Connects loss function solution to SVD of pointwise mutual information matrix
- Enables privacy-preserving algorithm design

### Theoretical
- Demonstrates effectiveness of multimodal learning vs single-modality learning
- Provides theoretical guarantees for representation quality

### Empirical
- Simulation studies across multiple configurations
- Real-world EHR data validation

## Key Insight
Many EHR studies concentrate on individual modalities (codes OR notes) or merge them rudimentarily. Structured codes and clinical notes contain "clinically relevant, inextricably linked and complementary health information" that should be analyzed jointly.

## Key Takeaways for Our Project
- Contrastive learning is theoretically grounded for EHR multimodal analysis
- SVD connection provides interpretability and privacy benefits
- Can be adapted as a pretraining objective before downstream task fine-tuning
- Privacy-preserving aspects relevant for clinical deployment
