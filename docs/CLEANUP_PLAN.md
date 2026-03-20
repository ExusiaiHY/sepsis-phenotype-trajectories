# Workspace Cleanup Plan

Generated: 2026-03-18
Status: EXECUTED (Phase 1-3 complete, 2026-03-18 19:45)

## Audit Summary

| Directory | Size | Purpose | Status |
|-----------|------|---------|--------|
| `src/` | 220KB | V1 original code (13 Python files) | **KEEP** as legacy reference |
| `s0/` | 12KB | S0 data layer (accepted, validated) | **KEEP** — active |
| `s1/` | 16KB | S1 encoder code | **KEEP** — active |
| `scripts/` | 28KB | Pipeline entry points (S0, S1, validation) | **KEEP** — active |
| `config/` | 8KB | Configs (V1, S0, S1) | **KEEP** |
| `tests/` | 4KB | S0 unit tests | **KEEP** |
| `docs/` | 2.1MB | Papers, logs, references | **KEEP** — reorganize |
| `multimodal/` | ~450MB | Multimodal fusion experiments (synthetic notes) | **ARCHIVE** — exploratory work, not on main research path |
| `data/external/` | ~20MB | Raw PhysioNet 2012 + Outcomes files | **KEEP** — read-only source |
| `data/processed/` | 557MB | V1 legacy processed data (proxy mortality) | **ARCHIVE** — superseded by data/s0/ |
| `data/s0/` | 220MB | S0 processed data (real mortality) | **KEEP** — active |
| `data/s1/` | 10MB | S1 embeddings + checkpoints | **KEEP** — active |
| `data/demo/` | 0B | Empty | **DELETE** |
| `data/raw/` | 0B | Empty | **DELETE** |
| `data/external/sepsis2019/` | ~30MB | Sepsis 2019 data (zip files are 0-byte stubs) | **ARCHIVE** |
| `db/` | 19MB | MIMIC-IV mock DuckDB database | **ARCHIVE** |
| `eicu-code-main/` | ~15MB | eICU reference repo (extracted from zip) | **ARCHIVE** — zip preserved |
| `mimic-code-main/` | ~7MB | MIMIC reference repo (extracted from zip) | **ARCHIVE** — zip preserved |
| `mimic-iv-data/` | small | MIMIC-IV data directory | **ARCHIVE** |
| `mimic-iv-mock/` | small | MIMIC-IV mock data | **ARCHIVE** |
| `eicu-code-main.zip` | 15MB | Source zip | **KEEP** |
| `mimic-code-main.zip` | 7.4MB | Source zip | **KEEP** |
| `outputs/` | 2MB | V1 figures + reports (proxy mortality) | **ARCHIVE** |
| `.venv/` | variable | Python virtual environment | **KEEP** (do not touch) |
| `__pycache__/` dirs | various | Python bytecode caches | **DELETE** (safe) |

## Proposed Actions

### Phase 1: Safe lossless moves (no deletions of content)

```bash
# 1. Create archive directory
mkdir -p archive/

# 2. Archive multimodal/ (synthetic notes experiments, not on main path)
mv multimodal/ archive/multimodal_fusion_experiments/

# 3. Archive V1 legacy processed data (superseded by S0)
mv data/processed/ archive/v1_processed_data/

# 4. Archive outputs/ (V1 results with proxy mortality)
mv outputs/ archive/v1_outputs/

# 5. Archive reference repos (zips preserved)
mv eicu-code-main/ archive/
mv mimic-code-main/ archive/
mv mimic-iv-data/ archive/
mv mimic-iv-mock/ archive/

# 6. Archive db/
mv db/ archive/

# 7. Archive sepsis2019 stubs
mv data/external/sepsis2019/ archive/sepsis2019_stubs/
```

### Phase 2: Delete empty directories and caches

```bash
# Safe deletions
rmdir data/demo/ data/raw/
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Remove LaTeX build artifacts from docs/
rm -f docs/RESEARCH_PAPER.aux docs/RESEARCH_PAPER.log docs/RESEARCH_PAPER.out
```

### Phase 3: Reorganize docs/

```
docs/
├── WORKLOG.md                    # Persistent work log
├── NEXT_STEPS.md                 # Next steps tracker
├── DECISIONS.md                  # Design decisions log
├── CLEANUP_PLAN.md               # This file
├── EXPERIMENT_REGISTRY.md        # All experiments
├── MANUSCRIPT_PATCHLIST.md       # Manuscript revision tracker
├── DEVELOPMENT_LOG.md            # V1 development history (keep for reference)
├── RESEARCH_PAPER.md             # Current manuscript (needs revision)
├── RESEARCH_PAPER.pdf            # PDF render
├── RESEARCH_PAPER.tex            # LaTeX source
├── references/                   # Literature files (13 papers)
├── midterm_progress_report.md    # Course deliverable
├── project_proposal.md           # Course deliverable
├── software_evaluation.md        # Course deliverable
├── software_evaluation.pdf
├── software_evaluation.tex
├── user_manual.md
├── user_manual.pdf
├── user_manual.tex
└── s11263-024-02032-8.pdf        # Downloaded reference paper
```

### Phase 4: Resulting clean structure

```
project/
├── src/                 # V1 legacy code (untouched, reference only)
├── s0/                  # S0 data layer (active)
├── s1/                  # S1 self-supervised encoder (active)
├── s15/                 # S1.5 (to be created)
├── scripts/             # Pipeline scripts
├── config/              # All configs
├── tests/               # Unit tests
├── data/
│   ├── external/        # Raw PhysioNet 2012 (read-only)
│   ├── s0/              # S0 processed outputs
│   └── s1/              # S1 embeddings + checkpoints
├── docs/                # All documentation and logs
├── archive/             # Safely moved obsolete materials
│   ├── multimodal_fusion_experiments/
│   ├── v1_processed_data/
│   ├── v1_outputs/
│   ├── eicu-code-main/
│   ├── mimic-code-main/
│   ├── mimic-iv-data/
│   ├── mimic-iv-mock/
│   ├── db/
│   └── sepsis2019_stubs/
├── eicu-code-main.zip   # Reference zip (kept)
├── mimic-code-main.zip  # Reference zip (kept)
├── requirements.txt
└── README.md

Estimated space savings: ~500MB moved to archive/, ~0B permanently deleted (except caches)
```

## Approval Required

The following actions require explicit user approval:
1. Move multimodal/ to archive/ (contains ~450MB of experiment artifacts)
2. Move data/processed/ to archive/ (contains 557MB V1 legacy data)
3. Move outputs/ to archive/ (contains V1 figures with proxy mortality)
4. Move reference repos to archive/
5. Delete empty directories and __pycache__
6. Delete LaTeX build artifacts

None of these are permanent deletions. Everything is moved to archive/.
