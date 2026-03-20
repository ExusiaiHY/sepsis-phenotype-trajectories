# CareBench: When Does Multimodal Learning Help in Healthcare?

**Source**: arXiv 2602.23614 (2025)
**URL**: https://arxiv.org/html/2602.23614
**Authors**: Kejing Yin et al., Hong Kong Baptist University
**Relevance**: Primary benchmark paper for EHR + CXR multimodal fusion

---

## Overview

CareBench is a comprehensive benchmarking study examining multimodal fusion of Electronic Health Records (EHR) and chest X-rays (CXR) for clinical prediction tasks. It addresses fundamental gaps in understanding when multimodal learning genuinely improves healthcare outcomes.

## Key Research Questions

1. **When multimodal fusion benefits medical tasks** - Under what conditions does combining modalities improve performance?
2. **Fusion strategy effectiveness** - Which architectural approaches work best with complete data?
3. **Robustness to missing modalities** - How do methods handle realistic missing data scenarios?
4. **Algorithmic fairness** - Do multimodal models maintain equitable performance across demographic groups?

## Dataset and Methodology

The benchmark leverages MIMIC-IV and MIMIC-CXR datasets:
- **Base cohort**: 26,947 ICU stays with temporal consistency requirements
- **Matched subset**: 7,149 ICU stays with paired EHR and chest radiographs

Three downstream prediction tasks:
- Phenotyping classification (25 disease categories)
- Mortality prediction (48-hour window)
- Length-of-stay prediction (remaining hospital duration)

## Models Benchmarked (17 total)

### Unimodal baselines
- LSTM, Transformer (EHR)
- ResNet-50 (CXR)

### Complete-modality methods
- Late Fusion, UTDE, DAFT, MMTM, AUG, InfoReg

### Missing-modality methods
- HEALNet, Flex-MoE, DrFuse, UMSE, ShaSpec, M3Care, MedFuse, SMIL

## Critical Findings

### Finding 1: Conditional Performance Gains
Multimodal fusion improves performance when modalities are complete, with gains concentrating in diseases requiring complementary information (e.g., congestive heart failure, COPD).

### Finding 2: Cross-Modal Learning Significance
Advanced fusion mechanisms (DrFuse, InfoReg, UTDE) substantially outperform naive concatenation by capturing clinically meaningful cross-modal dependencies.

### Finding 3: Modality Imbalance Challenge
EHR's temporal richness creates inherent information asymmetry against single-timepoint CXR. Methods explicitly addressing imbalance (AUG, InfoReg) outperform architecturally complex alternatives.

### Finding 4: Dramatic Degradation Under Missingness
In realistic settings where ~75% of ICU stays lack paired radiographs, multimodal benefits rapidly degrade unless models are explicitly designed for incomplete inputs. MedFuse and M3Care substantially outperform complete-data methods under missingness.

### Finding 5: Fairness Paradox
Multimodal fusion does not inherently improve fairness; some high-performing methods actually exacerbate subgroup disparities via unequal sensitivity.

## Key Takeaways for Our Project
- Late fusion is a surprisingly strong baseline
- Missing modality robustness is critical for clinical deployment
- Fairness evaluation is mandatory, not optional
- EHR temporal richness dominates over static imaging in most tasks
