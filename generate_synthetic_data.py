"""Generate a fully synthetic workbook with the schema expected by src.prepare_data."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SUBJECTS = ["MATEMÁTICA", "LÍNGUA PORTUGUESA", "CIÊNCIAS", "ENSINO RELIGIOSO"]

def generate(output, students_per_year=3000, seed=2026):
      rng = np.random.default_rng(seed)
      student_rows, grade_rows, payment_rows = [], [], []
      ra_counter = 100000
      for year in (2023, 2024, 2025):
                for _ in range(students_per_year):
                              ra_counter += 1
                              school = int(rng.choice([101, 102, 103, 104, 105, 106]))
                              upper = bool(rng.random() < 0.32)
                              series = int(rng.integers(1, 4 if upper else 10))
                              course = "ENSINO MÉDIO" if upper else "ENSINO FUNDAMENTAL"
                              new = rng.random() < 0.18
                              entry_year = year if new else int(rng.integers(max(2015, year - 8), year + 1))
                              ability = rng.normal(7.1, 1.25)
                              decline = max(0, rng.normal(0.25, 0.60))
                              arrears = int(rng.binomial(8, 0.055))
                              risk_logit = -4.15 + 0.42 * arrears + 0.42 * max(0, 6.2 - ability) + 0.30 * decline + 0.22 * new
                              cancelled = rng.random() < 1 / (1 + np.exp(-risk_logit))
                              # Most cancellations are after August, keeping a small number of pre-cutoff cases.
                              pre_cutoff = bool(cancelled and rng.random() < 0.12)
                              student_rows.append({
                                  "RA": ra_counter, "ANO": year, "FACULDADE": school,
                                  "SITUACAO": "Cancelado" if cancelled else "Ativo",
                                  "Curso_nome": course, "Série": series,
                                  "Turno": rng.choice(["MATUTINO", "VESPERTINO"], p=[0.68, 0.32]),
                                  "Novato": "S" if new else "N",
                                  "Data da Matrícula": pd.Timestamp(year, int(rng.integers(1, 4)), int(rng.integers(1, 25))),
                                  "Tempo_integral": "S" if rng.random() < 0.14 else "N",
                                  "Pastoral": "S" if rng.random() < 0.28 else "N",
                                  "Ano Ingresso": entry_year,
                                  "Autorização uso de imagem": np.nan if rng.random() < 0.09 else "SIM",
                              })
                              gr = {"RA": ra_counter, "ANO": year}
                              for subject in SUBJECTS:
                                                offset = rng.normal(0, 0.45)
                                                t1 = np.clip(ability + offset + rng.normal(0, 0.45), 0, 10)
                                                t2 = np.clip(t1 - decline + rng.normal(0, 0.45), 0, 10)
                                                if pre_cutoff:
                                                                      t2 = np.nan
                                                                  gr[f"{subject} - ETAPA1"] = round(float(t1), 2)
                                                gr[f"{subject} - ETAPA2"] = np.nan if pd.isna(t2) else round(float(t2), 2)
                                            grade_rows.append(gr)
                              last_billed = int(rng.integers(4, 9)) if pre_cutoff else 12
                              pay = {"RA": ra_counter, "Ano": year}
                              late_months = set(rng.choice(np.arange(1, 9), size=min(arrears, 8), replace=False).tolist())
                              for month in range(1, 13):
                                                pay[f"PARCELA_{month}"] = (1 if month in late_months else 0) if month <= last_billed else np.nan
                                            payment_rows.append(pay)
                      output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
              pd.DataFrame(student_rows).to_excel(writer, sheet_name="ALUNOS", index=False)
              pd.DataFrame(grade_rows).to_excel(writer, sheet_name="notas-pivot", index=False)
              pd.DataFrame(payment_rows).to_excel(writer, sheet_name="PAGAMENTOS", index=False)
          print(f"Synthetic workbook written to {output}")

if __name__ == "__main__":
      p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic_data.xlsx")
    p.add_argument("--students-per-year", type=int, default=3000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    generate(a.output, a.students_per_year, a.seed)
