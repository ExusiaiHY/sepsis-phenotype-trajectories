# MINGLE: Multimodal Fusion of EHR Structures and Semantics

**Source**: Studies in Health Technology and Informatics, 2025
**URL**: https://pubmed.ncbi.nlm.nih.gov/40775959/
**Authors**: Hejie Cui et al. (Stanford, Emory)
**DOI**: 10.3233/SHTI250941
**Relevance**: State-of-the-art EHR + clinical notes fusion using hypergraph neural networks

---

## Overview

MINGLE addresses integrating structural (tabular) and unstructured (textual notes) EHR data using a two-level semantic infusion strategy with hypergraph neural networks and LLMs.

## Key Challenge
EHRs contain heterogeneous information: structural data in tabular form and unstructured data in textual notes. Most existing methods handle these modalities separately or merge them crudely.

## Proposed Solution
Two-level infusion strategy:
1. **Medical concept semantics** - Embed structured clinical codes with semantic meaning
2. **Clinical note semantics** - Extract and align textual note information

Both levels are infused into hypergraph neural networks for joint representation learning.

## Results
- **11.83% relative improvement** in predictive performance
- Validated on MIMIC-III and private CRADLE dataset
- Demonstrates effective semantic integration for multimodal EHR fusion

## Key Takeaways for Our Project
- Hypergraph networks can capture higher-order relationships in clinical data
- Semantic alignment between structured codes and free text is valuable
- LLM-derived embeddings enhance clinical note representation
- Both public (MIMIC-III) and private datasets used for validation
