# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

This is a research project: **Self-Supervised Temporal Phenotype Trajectory Analysis of ICU Sepsis Patients**. The project uses the PhysioNet 2012 multi-center ICU database (11,986 patients) and progresses through stages S0–S3.

## Git Workflow — MANDATORY

**After completing each task, you MUST commit and push to GitHub.**

### When to commit

- After completing any requested code change
- After completing any experiment run
- After updating the manuscript (.tex, .md)
- After updating documentation (WORKLOG.md, DECISIONS.md, etc.)
- After applying manuscript patches
- After any file creation or deletion

### How to commit

1. `git add` only the relevant changed files (never use `git add -A` blindly — check `git status` first)
2. Write a clear commit message in this format:
   ```
   <type>(<scope>): <summary>

   <optional body with details>
   ```
   Types: `feat`, `fix`, `docs`, `refactor`, `experiment`, `manuscript`, `chore`
   Examples:
   - `experiment(s3): cross-center validation — 6/6 criteria passed`
   - `manuscript(p013): apply calibrated S3 wording`
   - `docs: update WORKLOG with S2-light results`
3. `git push origin main`

### What NOT to commit

- `.npy` files (large arrays — excluded by .gitignore)
- `.pt` checkpoint files
- `archive/` directory
- `__pycache__/`, `.venv/`

## Wording Policy (Manuscript)

When editing RESEARCH_PAPER.tex or RESEARCH_PAPER.md:
- USE: "descriptive temporal phenotype trajectories"
- USE: "cross-center temporal validation within the PhysioNet 2012 multi-center cohort"
- DO NOT USE: "robust cross-center generalization"
- DO NOT USE: "independent hospital generalization"
- DO NOT USE: "external validation" for our own results
- DO NOT USE: "full dynamic phenotyping"
- All mortality numbers must use real outcomes (14.2% base rate), never proxy labels

## Key File Locations

- Manuscript: `docs/RESEARCH_PAPER.tex`, `docs/RESEARCH_PAPER.md`
- Logs: `docs/WORKLOG.md`, `docs/DECISIONS.md`, `docs/EXPERIMENT_REGISTRY.md`
- Patches: `docs/MANUSCRIPT_PATCHLIST.md`
- Data pipeline: `s0/`, `s1/`, `s15/`, `s2light/`
- Scripts: `scripts/`
- Config: `config/`

## Running Commands

- Always prefix Python commands with: `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`
- LaTeX compilation: `cd docs && pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex` (run twice)
- Python version: use `python3` or `python3.14`
