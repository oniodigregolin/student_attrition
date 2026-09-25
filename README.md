# Synthetic student-attrition pipeline

Public, anonymized mirror of the analysis pipeline. The repository contains no original student records. The included workbook is generated artificially and is intended to validate execution and illustrate the expected schema. Results obtained from it are demonstrations and must not be interpreted as empirical findings from the study.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate the synthetic workbook

```bash
python generate_synthetic_data.py --students-per-year 3000 --seed 2026
```

This creates `data/synthetic_data.xlsx` with the three sheets expected by the pipeline: `ALUNOS`, `notas-pivot`, and `PAGAMENTOS`.

## Run

Quick schema and preparation check:

```bash
python -m src.prepare_data --data-file data/synthetic_data.xlsx --output-dir outputs
```

Complete analysis:

```bash
python -m scripts.run_pipeline --data-file data/synthetic_data.xlsx --output-dir outputs
```

Faster smoke test for model fitting:

```bash
python -m src.fit_models --n-settings 2 --n-jobs 2 --output-dir outputs
python -m src.build_results --output-dir outputs
```

## Data availability

The original administrative data are not distributed. The synthetic workbook does not reproduce real individuals, identifiers, schools, or observed study results. It preserves only the input schema and broad logical relationships required to exercise the code.
