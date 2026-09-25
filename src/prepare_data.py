"""Build the analytic file: one row per student-year still enrolled at the cut-off.

Cut-off: end of the 2nd assessment stage (late August).
Outcome: enrolment cancelled after the cut-off (September to December).

    python -m src.prepare_data
    python -m src.prepare_data --grade-method rank   # sensitivity analysis, separate file
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.reproducibility import setup_logging

log = logging.getLogger("prepare")

SHEETS = {"students": "ALUNOS", "grades": "notas-pivot", "payments": "PAGAMENTOS"}
SOURCE_SCHOOL_COLUMN = "FACULDADE"   # school identifier as named in the source system
SCHOOL_COLUMN = "SCHOOL_ID"
REQUIRED = {
    "ALUNOS": ["RA", "ANO", SOURCE_SCHOOL_COLUMN, "SITUACAO", "Curso_nome", "Série", "Turno",
               "Novato", "Data da Matrícula", "Tempo_integral", "Pastoral", "Ano Ingresso",
               "Autorização uso de imagem"],
    "notas-pivot": ["RA", "ANO"],
    "PAGAMENTOS": ["RA", "Ano"] + [f"PARCELA_{i}" for i in range(1, config.N_INSTALMENTS + 1)],
}
RELIGION_SUBJECTS = ("ENSINO RELIGIOSO", "CULTURA RELIGIOSA")


class SchemaError(ValueError):
    pass


def _fail(path, sheet, column, problem, action):
    raise SchemaError(f"{path} | sheet {sheet} | column {column}: {problem}. {action}")


def load_raw_data(path):
    path = Path(path)
    if not path.exists():
        _fail(path, "-", "-", "file not found", "Place the original spreadsheet at data/ (see data/README.md)")
    log.info("Reading %s", path.name)
    book = pd.read_excel(path, sheet_name=None)
    missing = [s for s in SHEETS.values() if s not in book]
    if missing:
        _fail(path, ", ".join(missing), "-", "sheet missing", "Check that the export includes all sheets")
    return {key: clean_column_names(book[sheet]) for key, sheet in SHEETS.items()}


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_source_schema(raw, path=config.DATA_FILE):
    for key, sheet in SHEETS.items():
        df = raw[key]
        for col in REQUIRED[sheet]:
            if col not in df.columns:
                _fail(path, sheet, col, "column missing", "Re-export the sheet with the original column names")
    grade_cols = [c for c in raw["grades"].columns if "ETAPA" in c]
    if not grade_cols:
        _fail(path, "notas-pivot", "* - ETAPAn", "no grade columns found", "Grade columns must end in ' - ETAPA1', ' - ETAPA2', ...")
    for key, sheet in SHEETS.items():
        year_col = "Ano" if sheet == "PAGAMENTOS" else "ANO"
        years = pd.to_numeric(raw[key][year_col], errors="coerce")
        if years.isna().any():
            _fail(path, sheet, year_col, f"{int(years.isna().sum())} non-numeric years", "Fix or remove these rows")
        if pd.to_numeric(raw[key]["RA"], errors="coerce").isna().any():
            _fail(path, sheet, "RA", "missing or non-numeric identifiers", "Every row needs a numeric RA")


def rename_and_convert(raw):
    """Rename the school identifier and set explicit types."""
    students = raw["students"].rename(columns={SOURCE_SCHOOL_COLUMN: SCHOOL_COLUMN})
    grades, payments = raw["grades"].copy(), raw["payments"].copy()

    for df in (students, grades, payments):
        df["RA"] = pd.to_numeric(df["RA"]).astype("int64")
    students["ANO"] = pd.to_numeric(students["ANO"]).astype("int64")
    grades["ANO"] = pd.to_numeric(grades["ANO"]).astype("int64")
    payments["Ano"] = pd.to_numeric(payments["Ano"]).astype("int64")
    students[SCHOOL_COLUMN] = pd.to_numeric(students[SCHOOL_COLUMN]).astype("int64")
    students["Série"] = pd.to_numeric(students["Série"]).astype("int64")
    students["Ano Ingresso"] = pd.to_numeric(students["Ano Ingresso"]).astype("int64")

    dates = pd.to_datetime(students["Data da Matrícula"], errors="coerce")
    bad = dates.isna() & students["Data da Matrícula"].notna()
    if bad.any():
        log.warning("%d enrolment dates could not be parsed and were set to missing", bad.sum())
    students["Data da Matrícula"] = dates

    grade_cols = [c for c in grades.columns if "ETAPA" in c]
    before = grades[grade_cols].notna().sum().sum()
    grades[grade_cols] = grades[grade_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    lost = before - grades[grade_cols].notna().sum().sum()
    if lost:
        log.warning("%d non-numeric grade values set to missing", lost)
    if (grades[grade_cols] < 0).any().any():
        log.warning("Negative grades found; they are kept as recorded")

    inst = [f"PARCELA_{i}" for i in range(1, config.N_INSTALMENTS + 1)]
    payments[inst] = payments[inst].apply(pd.to_numeric, errors="coerce").astype(float)
    return students, grades, payments


def select_eligible_students(students):
    """Regular primary (EF) and secondary (EM) courses, minus the excluded schools."""
    def segment(course):
        if "ENSINO MÉDIO" in course:
            return "EM"
        if "ENSINO FUNDAMENTAL" in course:
            return "EF"
        return None   # early childhood, language courses, etc.

    students = students.copy()
    students["SEG"] = students["Curso_nome"].astype(str).apply(segment)
    students = students[students["SEG"].notna()]
    students = students[~students[SCHOOL_COLUMN].isin(config.EXCLUDED_SCHOOLS)]
    return students.drop_duplicates(["RA", "ANO"], keep="first")


def reshape_grades(grades, students):
    cols = [c for c in grades.columns if "ETAPA" in c]
    long = grades.melt(id_vars=["RA", "ANO"], value_vars=cols, var_name="col").dropna()
    long[["subject", "stage"]] = long["col"].str.rsplit(" - ", n=1, expand=True)
    long["stage"] = long["stage"].str[-1].astype(int)
    long = long[long["stage"] <= config.CUTOFF_STAGE]   # later stages are not known at the cut-off
    return long.merge(students[["RA", "ANO", SCHOOL_COLUMN]], on=["RA", "ANO"], how="inner")


def rescale_grade_group(values, method="p995"):
    """Fix misplaced decimal points and put one school/year/subject/stage group on 0-10.

    Schools use different scales (0-10, 0-30, 0-100, ...) and many grades were typed without
    the decimal point (2950 for 29.50). Values far above the usual maximum of the group are
    divided by 10 (up to three times); the group is then rescaled so that its 99.5th
    percentile maps to 10. ``method="rank"`` replaces the last step by percentile ranks and
    is only used as a sensitivity analysis.
    """
    v = np.asarray(values, dtype=float).copy()
    median = np.median(v)
    typical = v[v <= 3.5 * max(median, 1)]
    upper = np.quantile(typical, 0.995) if len(typical) else v.max()
    upper = max(upper, 1e-9)

    divided = np.zeros(len(v), dtype=bool)
    for _ in range(3):
        too_big = v > upper * 1.05
        divided |= too_big
        v[too_big] = v[too_big] / 10

    if method == "rank":
        scaled = pd.Series(v).rank(pct=True).to_numpy() * 10
        truncated = np.zeros(len(v), dtype=bool)
    else:
        top = np.quantile(v, 0.995)
        raw_scaled = v / max(top, 1e-9) * 10
        truncated = (raw_scaled > 10) | (raw_scaled < 0)
        scaled = np.clip(raw_scaled, 0, 10)
    info = {"n": len(v), "n_divided": int(divided.sum()), "n_truncated": int(truncated.sum()),
            "p50_before": float(np.median(values)), "p995_before": float(np.quantile(values, 0.995)),
            "p50_after": float(np.median(scaled)), "p995_after": float(np.quantile(scaled, 0.995)),
            "no_variation": bool(np.ptp(np.asarray(values, dtype=float)) == 0)}
    return scaled, info


def build_grade_features(long, method="p995"):
    long = long.copy()
    long["grade"] = np.nan
    audit = []
    for key, idx in long.groupby([SCHOOL_COLUMN, "ANO", "subject", "stage"]).indices.items():
        scaled, info = rescale_grade_group(long["value"].to_numpy()[idx], method)
        long.iloc[idx, long.columns.get_loc("grade")] = scaled
        audit.append(dict(zip([SCHOOL_COLUMN, "ANO", "subject", "stage"], key)) | info)

    means = long.pivot_table(index=["RA", "ANO"], columns="stage", values="grade", aggfunc="mean")
    means = means.reindex(columns=[1, 2])
    means.columns = ["T1_MEAN", "T2_MEAN"]

    stage2 = long[long["stage"] == 2].assign(below6=lambda d: d["grade"] < 6)
    below6 = stage2.groupby(["RA", "ANO"])["below6"].sum().rename("T2_BELOW6")

    religion = long[long["subject"].str.startswith(RELIGION_SUBJECTS)]
    religion = religion.pivot_table(index=["RA", "ANO"], columns="stage", values="grade", aggfunc="mean")
    rel_decline = (religion[1] - religion[2]).rename("REL_DECLINE")

    features = means.join(below6).join(rel_decline).reset_index()
    return features, pd.DataFrame(audit)


def build_payment_features(payments):
    inst = [f"PARCELA_{i}" for i in range(1, config.N_INSTALMENTS + 1)]
    billed = payments[inst].notna().to_numpy()
    last_billed = np.where(billed.any(axis=1), config.N_INSTALMENTS - billed[:, ::-1].argmax(axis=1), 0)
    arrears = (payments[inst[:config.LAST_INSTALMENT_BEFORE_CUTOFF]] == 1).sum(axis=1).to_numpy()
    pay = pd.DataFrame({"RA": payments["RA"], "ANO": payments["Ano"],
                        "LAST_BILLED": last_billed, "ARREARS_TO_AUG": arrears})
    return pay.drop_duplicates(["RA", "ANO"])


def apply_prediction_cutoff(df):
    """Keep students still enrolled at the end of stage 2.

    LAST_BILLED comes from the whole-year billing record, i.e. partly from after the
    cut-off. It is used only to decide who had already left by August and never enters
    the predictors.
    """
    left_before = df["T2_MEAN"].isna() | (df["LAST_BILLED"].fillna(config.N_INSTALMENTS)
                                          <= config.LAST_INSTALMENT_BEFORE_CUTOFF)
    counts = pd.DataFrame({"with_grades": df.groupby("ANO").size(),
                           "departures_in_year": df.groupby("ANO")["y"].sum(),
                           "departures_before_cutoff": df[left_before].groupby("ANO")["y"].sum()})
    return df[~left_before].copy(), counts


def calculate_historical_school_exit_rate(students, df, past_only=False):
    """Cancellation rate of each school in the training years other than the row's year.

    2023 rows use 2024, 2024 rows use 2023 and 2025 rows use 2023-2024; the test year is
    never used. With ``past_only=True`` only earlier years are used (2023 has no history).
    Schools without usable years get NaN, imputed later inside the model pipeline.
    """
    s = students.assign(cancelled=(students["SITUACAO"] == "Cancelado").astype(int))
    rates = s[s["ANO"] < config.TEST_YEAR].groupby([SCHOOL_COLUMN, "ANO"])["cancelled"].agg(["sum", "size"])
    known = set(rates.index.get_level_values(0))
    table = {}
    for school in df[SCHOOL_COLUMN].unique():
        for year in df["ANO"].unique():
            if school not in known:
                table[(school, year)] = (np.nan, "")
                continue
            r = rates.loc[school]
            r = r[r.index < year] if past_only else r[r.index != year]
            size = r["size"].sum()
            rate = r["sum"].sum() / size if size > 0 else np.nan
            table[(school, year)] = (rate, ",".join(str(y) for y in r.index))
    values = [table[(s, y)][0] for s, y in zip(df[SCHOOL_COLUMN], df["ANO"])]
    audit = pd.DataFrame([{SCHOOL_COLUMN: k[0], "ANO": k[1], "rate": v[0], "years_used": v[1]}
                          for k, v in table.items()])
    return np.array(values, dtype=float), audit


def build_predictors(df, students):
    df = df.copy()
    # class = school x year x grade level
    df["CLASS"] = (df[SCHOOL_COLUMN].astype(str) + "_" + df["ANO"].astype(str) + "_"
                   + df["SEG"] + df["Série"].astype(str))
    by_class = df.groupby("CLASS")
    df["CLASS_SIZE"] = by_class["RA"].transform("size")
    df["PCT_T1"] = by_class["T1_MEAN"].rank(pct=True)
    df["PCT_T2"] = by_class["T2_MEAN"].rank(pct=True)
    df["DECLINE_PTS"] = df["T1_MEAN"] - df["T2_MEAN"]
    df["DECLINE_PCT"] = df["PCT_T1"] - df["PCT_T2"]
    sd = by_class["DECLINE_PTS"].transform("std").replace(0, np.nan)
    df["DECLINE_Z"] = ((df["DECLINE_PTS"] - by_class["DECLINE_PTS"].transform("mean")) / sd).fillna(0)

    df["NEW"] = (df["Novato"] == "S").astype(int)
    df["YEARS_NET"] = (df["ANO"] - df["Ano Ingresso"]).clip(lower=0)
    df["UPPER_SEC"] = (df["SEG"] == "EM").astype(int)
    df["AFTERNOON"] = (df["Turno"] == "VESPERTINO").astype(int)
    df["FULLTIME"] = (df["Tempo_integral"] == "S").astype(int)
    df["ENROL_MONTH"] = df["Data da Matrícula"].dt.month
    df["NO_IMAGE_CONSENT"] = df["Autorização uso de imagem"].isna().astype(int)
    df["CHAPLAINCY"] = (df["Pastoral"] == "S").astype(int)
    df["ARREARS_TO_AUG"] = df["ARREARS_TO_AUG"].fillna(0)

    df["SCHOOL_EXIT_RATE"], rate_audit = calculate_historical_school_exit_rate(students, df)
    df["SCHOOL_EXIT_RATE_PAST"], _ = calculate_historical_school_exit_rate(students, df, past_only=True)

    return df, rate_audit


def validate_analytic_dataset(df):
    problems = []
    if df.duplicated(["RA", "ANO"]).any():
        problems.append("duplicated RA/ANO rows")
    if sorted(df["ANO"].unique()) != config.YEARS:
        problems.append(f"unexpected years {sorted(df['ANO'].unique())}")
    if SCHOOL_COLUMN not in df.columns:
        problems.append("SCHOOL_ID missing")
    forbidden = set(config.FEATURES) & set(config.COHORT_COLUMNS + config.AUXILIARY_COLUMNS + config.ID_COLUMNS + ["y"])
    if forbidden:
        problems.append(f"non-predictor columns listed as features: {sorted(forbidden)}")
    missing = [f for f in config.FEATURES if f not in df.columns]
    if missing:
        problems.append(f"features missing: {missing}")
    if any(c in df.columns for c in config.COHORT_COLUMNS):
        problems.append("cohort-only columns must not be saved in the analytic file")
    X = df[[f for f in config.FEATURES if f in df.columns]]
    if np.isinf(X.to_numpy(dtype=float)).any():
        problems.append("infinite values in the predictors")
    ranges = {"T1_MEAN": (0, 10), "T2_MEAN": (0, 10), "PCT_T2": (0, 1), "SCHOOL_EXIT_RATE": (0, 1)}
    for col, (lo, hi) in ranges.items():
        if col in df and ((df[col] < lo) | (df[col] > hi)).any():
            problems.append(f"{col} outside [{lo}, {hi}]")
    if problems:
        raise ValueError("analytic dataset failed validation: " + "; ".join(problems))

    summary = df.groupby("ANO")["y"].agg(students="size", departures="sum", prevalence="mean")
    for year, row in summary.iterrows():
        log.info("%d: %d students, %d departures (%.2f%%)", year, row.students, row.departures, 100 * row.prevalence)
    n_missing = df[config.FEATURES].isna().sum()
    log.info("Missing values: %s", n_missing[n_missing > 0].to_dict())
    return summary


def save_analytic_dataset(df, path):
    cols = config.ID_COLUMNS + ["y"] + config.FEATURES + config.AUXILIARY_COLUMNS
    out = df[cols].reset_index(drop=True)
    validate_analytic_dataset(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    log.info("Saved %s", path)
    return out


def prepare(data_file=config.DATA_FILE, output_dir=config.OUTPUT_DIR, grade_method="p995"):
    raw = load_raw_data(data_file)
    validate_source_schema(raw, data_file)
    students, grades, payments = rename_and_convert(raw)
    students = select_eligible_students(students)

    grade_features, grade_audit = build_grade_features(reshape_grades(grades, students), grade_method)
    df = students.merge(grade_features, on=["RA", "ANO"], how="inner")
    df = df.merge(build_payment_features(payments), on=["RA", "ANO"], how="left")
    df["y"] = (df["SITUACAO"] == "Cancelado").astype(int)

    df, counts = apply_prediction_cutoff(df)
    df, rate_audit = build_predictors(df, students)

    suffix = "" if grade_method == "p995" else f"_grade-{grade_method}"
    out = save_analytic_dataset(df, Path(output_dir) / f"analytic{suffix}.pkl")

    audits = Path(output_dir) / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    grade_audit.to_csv(audits / f"grade_rescaling_audit{suffix}.csv", index=False)
    rate_audit.to_csv(audits / "school_exit_rate_audit.csv", index=False)
    counts["enrolled_at_cutoff"] = out.groupby("ANO").size()
    counts["departures_after_cutoff"] = out.groupby("ANO")["y"].sum()
    counts["prevalence"] = counts["departures_after_cutoff"] / counts["enrolled_at_cutoff"]
    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    counts.to_csv(tables / f"population_by_year{suffix}.csv")

    g = grade_audit
    log.info("Grades: %d values, %.2f%% divided by 10, %.2f%% truncated, %d groups with fewer than 10 values",
             g["n"].sum(), 100 * g["n_divided"].sum() / g["n"].sum(),
             100 * g["n_truncated"].sum() / g["n"].sum(), (g["n"] < 10).sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", default=str(config.DATA_FILE))
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--grade-method", choices=["p995", "rank"], default="p995")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(Path(args.output_dir) / "logs", args.log_level, "prepare")
    prepare(args.data_file, args.output_dir, args.grade_method)


if __name__ == "__main__":
    main()
