# MIMIC-Multimodal: DiReCT Benchmark for Foundation Models

**Source**: GitHub (nliulab/MIMIC-Multimodal)
**URL**: https://github.com/nliulab/MIMIC-Multimodal
**Authors**: Kunyu Yu, Nan Liu (Duke-NUS Medical School)
**Relevance**: Comprehensive benchmark for unimodal and multimodal foundation models on MIMIC-IV

---

## Overview

A comprehensive benchmark framework evaluating foundation models on electronic health records using MIMIC-IV, MIMIC-CXR, and MIMIC-IV-Note datasets. Systematically compared 12 foundation models as unimodal encoders and multimodal learners.

## Data Pipeline

### Processing Stages
1. **Master Dataset Generation** - Initial data compilation
2. **Benchmark Dataset Creation** - Standardized processing with configurable parameters:
   - Patient age thresholds (default: >= 18)
   - Temporal windows around ICU admission (start_diff, end_diff in hours)
3. **Outcome Enrichment** - Integrates invasive ventilation and sepsis onset metrics using official MIMIC Code Repository SQL

### Data Sources
- MIMIC-IV v2.2 (core clinical data)
- MIMIC-CXR (chest radiography images)
- MIMIC-IV-Note (clinical documentation)

## Model Architecture

### Unimodal Encoding
Modality-specific representations extracted using:
- **Demographics**: Direct feature engineering
- **Time-series**: Three embedding techniques (fixed intervals, GRU, moment-based)
- **Images**: CXR-Foundation and Swin Transformer
- **Clinical Notes**: OpenAI embeddings and RadBERT

Combined into unified representations for logistic regression-based prediction.

### Multimodal Learning
Vision-language models process patient profiles generated from ICU data + clinical imagery:
- GPT-4o-mini, LLaVA-v1.5-7B, LLaVA-Med
- Gemini 2.5-VL-7B, MedGemma-4B
- Qwen-based variants

## Clinical Tasks
- In-hospital mortality prediction
- Length-of-stay prediction (ICU stay > 3 days)

## Key Takeaways for Our Project
- Standardized data pipeline is critical for reproducibility
- Foundation model embeddings can serve as strong unimodal baselines
- Logistic regression on concatenated embeddings is a valid fusion baseline
- Official MIMIC-code SQL for outcome definitions ensures consistency
