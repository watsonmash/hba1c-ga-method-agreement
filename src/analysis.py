"""Method-agreement analysis of HbA1c and glycated albumin (GA%) in adults on
dolutegravir-based ART, with hypertension and Atherogenic Index of Plasma (AIP) models.

Usage:
    python src/analysis.py                       # runs on data/synthetic_study_data.csv
    python src/analysis.py path/to/your_data.xlsx  # real data (never commit it)
"""
import warnings, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
matplotlib_backend = mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

DATA_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic_study_data.csv")
OUT, FIG, TAB = Path("outputs"), Path("outputs/figures"), Path("outputs/tables")
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

ADA_PRE, ADA_DM = 5.7, 6.5
HB_HYPER = ADA_DM        # hyperglycaemia = HbA1c >= 6.5% (ADA diagnostic threshold)
GA_REF = (10.7, 15.2)      # Matsha reference interval, ref 54
GA_HYPER = 13.44           # Kengne, Youden-optimal for OGTT dysglycaemia (95% CI 12.72-14.71), ref 3
GA_HYPER_CI = (12.72, 14.71)
HTN_S, HTN_D = 140, 90
AIP_BANDS = [(-np.inf, 0.10, "Low"), (0.10, 0.24, "Medium"), (0.24, np.inf, "High")]
LIPID = {"TC": 5.2, "LDL": 3.4, "TG": 1.7, "HDL_M": 1.03, "HDL_F": 1.30}
FRIED_MAX = 4.5
ALPHA = 0.05
np.random.seed(42)

C_BLUE, C_ORANGE, C_GREEN = "#0072B2", "#D55E00", "#009E73"
CAT = [C_BLUE, C_ORANGE, C_GREEN]
INK, MUTED, GRIDC = "#1a1a19", "#5f5f5c", "#d8d8d4"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": INK,
    "axes.edgecolor": GRIDC, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "text.color": INK, "lines.linewidth": 2.0,
})

RESULTS = {}


SYNTHETIC = "synthetic" in DATA_FILE.name.lower()


def save_fig(fig, name):
    if SYNTHETIC:
        fig.text(0.99, 0.01, "SYNTHETIC DATA – illustrative only", ha="right", va="bottom",
                 fontsize=8, color="#C0392B", alpha=0.85)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def save_table(d, name):
    d.to_csv(TAB / f"{name}.csv", index=False)


def wilson(k, n, alpha=ALPHA):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (p * 100, max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def is_normal(s):
    s = pd.Series(s).dropna()
    return len(s) >= 3 and stats.shapiro(s)[1] > ALPHA


def summarise(s, dp=1):
    s = pd.Series(s).dropna()
    if is_normal(s):
        return f"{s.mean():.{dp}f} ({s.std():.{dp}f})", "Mean (SD)"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.{dp}f} ({q1:.{dp}f}–{q3:.{dp}f})", "Median (IQR)"


def cont_row(col, label, dp=1, data=None):
    d = df if data is None else data
    s = pd.Series(d[col]).dropna()
    summ, stat = summarise(s, dp)
    return {"Variable": label, "n": len(s), "Summary": summ, "Statistic": stat,
            "Range": f"{s.min():.{dp}f}–{s.max():.{dp}f}"}


def cat_rows(series, label, order=None, denom=None):
    s = pd.Series(series)
    n = len(s.dropna()) if denom is None else denom
    levels = order if order is not None else list(s.dropna().unique())
    out = [{"Variable": label, "n": "", "%": "", "95% CI": f"n = {n}"}]
    for lv in levels:
        k = int((s == lv).sum())
        p, lo, hi = wilson(k, n)
        out.append({"Variable": f"    {lv}", "n": k, "%": f"{p:.1f}",
                    "95% CI": f"{lo:.1f}–{hi:.1f}"})
    return out


def flag_row(mask, label, denom=None):
    m = pd.Series(mask).fillna(False).astype(bool)
    n = len(m) if denom is None else denom
    k = int(m.sum())
    p, lo, hi = wilson(k, n)
    return {"Variable": label, "n": k, "%": f"{p:.1f}", "95% CI": f"{lo:.1f}–{hi:.1f}"}


def blank():
    return {"Variable": "", "n": "", "%": "", "95% CI": ""}


def fmt_p(p):
    return "—" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


# ---------------------------------------------------------------- load
if DATA_FILE.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_FILE).dropna(how="all")
else:  # study workbook has a grouped header row above the column names
    df = pd.read_excel(DATA_FILE, header=1).dropna(how="all")
df.columns = [str(c).strip() for c in df.columns]
assert df["Study ID"].is_unique
N = len(df)
RESULTS["N"] = N

df["BMI"] = df["Weight (kg)"] / (df["Height (cm)"] / 100) ** 2
df["BMI_cat"] = pd.Categorical(
    pd.cut(df["BMI"], [0, 18.5, 25, 30, np.inf],
           labels=["Underweight", "Normal weight", "Overweight", "Obese"], right=False),
    ["Underweight", "Normal weight", "Overweight", "Obese"], ordered=True)
df["GA_pct"] = df["Glyc. Albumin (GA) g/L"] / df["Albumin (ALB) g/L"] * 100
df["AIP"] = np.log10(df["TG (mmol/L)"] / df["HDL (mmol/L)"])
df["GA_HbA1c"] = df["GA_pct"] / df["HbA1c (%)"]
df["nonHDL"] = df["TC (mmol/L)"] - df["HDL (mmol/L)"]
df["TC_HDL"] = df["TC (mmol/L)"] / df["HDL (mmol/L)"]
df["LDL_calc"] = df["TC (mmol/L)"] - df["HDL (mmol/L)"] - df["TG (mmol/L)"] / 2.2
df.loc[df["TG (mmol/L)"] > FRIED_MAX, "LDL_calc"] = np.nan


def parse_vl(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.startswith("<"):
        try:
            return float(s[1:])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


df["VL"] = df["Viral Load (copies/mL)"].apply(parse_vl)
df["VL_supp"] = np.where(df["VL"].isna(), np.nan, (df["VL"] < 50).astype(float))

cond = df["Chronic Condition(s)"].fillna("").astype(str).str.lower()
df["c_htn"] = cond.str.contains(r"hpt|hyperten|hypeten", regex=True)
df["c_asthma"] = cond.str.contains("asthma")
df["c_arthritis"] = cond.str.contains("arthritis")
df["c_stroke"] = cond.str.contains("stroke")
df["c_ulcer"] = cond.str.contains("ulcer")
df["has_chronic"] = df["Chronic Disease (Y/N)"].astype(str).str.strip().str.startswith("Y")

ANTIHTN = (r"nifedip|nefedip|amlodip|amylopid|enalapril|hct|htc|indapamide|losartan|"
           r"locatine|atenolol|bisoprolol|carvedilol|hydrochloroth")
df["on_antihtn"] = df["Chronic Disease Meds"].fillna("").astype(str).str.lower().str.contains(ANTIHTN, regex=True)
df["BP_high"] = (df["BP Systolic (mmHg)"] >= HTN_S) | (df["BP Diastolic (mmHg)"] >= HTN_D)
df["htn"] = (df["BP_high"] | df["on_antihtn"]).astype(int)

# Occupation free text is mapped to three groups by keyword (lists kept generic for the public repo)
FORMAL = ["administrator", "agronomist", "driver", "gardener", "general hand", "housemaid", "inspector",
          "marketing", "mining", "police", "printer", "receptionist", "sales", "security", "shop",
          "teacher", "technician", "engineer", "clerk", "nurse", "construction"]
SELF = ["carpenter", "hair", "smith", "seamstress", "self employed", "self-employed", "transporter",
        "vendor", "bar"]
NOTW = ["unemployed", "unemplyed", "retired", "housewife", "student"]


def employment(v):
    if pd.isna(v):
        return "Not recorded"
    s = str(v).strip().lower()
    if any(k == s or k in s for k in NOTW):
        return "Not in paid work"
    if any(k in s for k in SELF):
        return "Self-employed or informal work"
    if any(k in s for k in FORMAL):
        return "Formal employment"
    return "Not recorded"


df["Employment"] = df["Occupation"].apply(employment)
df["Income"] = pd.to_numeric(df["Avg Income"], errors="coerce")
df["Sleep"] = pd.to_numeric(df["Sleep (hrs/day)"], errors="coerce")
df["Stress"] = pd.to_numeric(df["Stress Level"], errors="coerce")

CATS = ["Normoglycaemia", "Prediabetes", "Diabetes"]


def hb_cat(v):
    if pd.isna(v):
        return np.nan
    return "Diabetes" if v >= ADA_DM else ("Prediabetes" if v >= ADA_PRE else "Normoglycaemia")


df["HbA1c_cat"] = pd.Categorical(df["HbA1c (%)"].apply(hb_cat), CATS, ordered=True)

df["HDL_low"] = np.where(df["Sex"].str.startswith("M"),
                         df["HDL (mmol/L)"] < LIPID["HDL_M"], df["HDL (mmol/L)"] < LIPID["HDL_F"])
df["TC_high"] = df["TC (mmol/L)"] > LIPID["TC"]
df["LDL_high"] = df["LDL (mmol/L)"] > LIPID["LDL"]
df["TG_high"] = df["TG (mmol/L)"] > LIPID["TG"]
df["any_dys"] = df[["TC_high", "LDL_high", "TG_high", "HDL_low"]].any(axis=1)


def aip_band(v):
    if pd.isna(v):
        return np.nan
    for lo, hi, nm in AIP_BANDS:
        if lo <= v < hi:
            return nm
    return np.nan


df["AIP_band"] = pd.Categorical(df["AIP"].apply(aip_band), ["Low", "Medium", "High"], ordered=True)

# ---------------------------------------------------------------- QC
flags = pd.DataFrame(index=df.index)
flags["Weight outside 30 to 200 kg"] = ~df["Weight (kg)"].between(30, 200)
flags["Body mass index outside 12 to 60 kg/m2"] = ~df["BMI"].between(12, 60)
flags["Serum albumin outside 15 to 55 g/L"] = ~df["Albumin (ALB) g/L"].between(15, 55)
flags["Glycated albumin outside 5 to 35%"] = ~df["GA_pct"].between(5, 35)
flags["HDL cholesterol outside 0.3 to 3.0 mmol/L"] = ~df["HDL (mmol/L)"].between(0.3, 3.0)
flags["Triglycerides above the Friedewald ceiling"] = df["TG (mmol/L)"] > FRIED_MAX
df["qc"] = flags.any(axis=1)

qc_rows = []
for col in flags.columns:
    for i in flags.index[flags[col]]:
        qc_rows.append({"Study ID": df.loc[i, "Study ID"], "Issue": col,
                        "Weight (kg)": round(df.loc[i, "Weight (kg)"], 1),
                        "BMI": round(df.loc[i, "BMI"], 1),
                        "Albumin (g/L)": df.loc[i, "Albumin (ALB) g/L"],
                        "GA (%)": round(df.loc[i, "GA_pct"], 2),
                        "HbA1c (%)": df.loc[i, "HbA1c (%)"],
                        "HDL (mmol/L)": df.loc[i, "HDL (mmol/L)"],
                        "TG (mmol/L)": df.loc[i, "TG (mmol/L)"]})
qc_report = pd.DataFrame(qc_rows).sort_values(["Study ID", "Issue"]).reset_index(drop=True)
save_table(qc_report, "t_qc")
RESULTS["qc_n"] = int(df["qc"].sum())

# ---------------------------------------------------------------- Table 1 socio
socio = []
socio += cat_rows(df["Sex"], "Sex", ["Female", "Male"], denom=N)
socio += cat_rows(pd.cut(df["Age (yrs)"], [0, 30, 40, 50, np.inf],
                         labels=["18–29", "30–39", "40–49", "50 and over"], right=False),
                  "Age band, years", ["18–29", "30–39", "40–49", "50 and over"], denom=N)
socio += cat_rows(df["Marital Status"], "Marital status",
                  ["Married", "Single", "Divorced", "Widowed"])
socio += cat_rows(df["Education Level"], "Highest level of education",
                  ["Primary", "Secondary", "Tertiary"], denom=N)
socio += cat_rows(df["Employment"], "Employment status",
                  ["Formal employment", "Self-employed or informal work",
                   "Not in paid work", "Not recorded"], denom=N)
save_table(pd.DataFrame(socio), "t1_socio")

save_table(pd.DataFrame([cont_row("Age (yrs)", "Age, years"),
                         cont_row("Income", "Reported average income")]), "t2_age_income")

# ---------------------------------------------------------------- anthro
save_table(pd.DataFrame([
    cont_row("Weight (kg)", "Weight, kg"),
    cont_row("Height (cm)", "Height, cm"),
    cont_row("BMI", "Body mass index, kg/m²"),
    cont_row("BP Systolic (mmHg)", "Systolic blood pressure, mmHg"),
    cont_row("BP Diastolic (mmHg)", "Diastolic blood pressure, mmHg"),
]), "t3_anthro")

bp = cat_rows(df["BMI_cat"], "Body mass index category",
              ["Underweight", "Normal weight", "Overweight", "Obese"], denom=N)
bp.append(flag_row(df["BMI"] >= 25, "Overweight or obese", denom=N))
bp.append(blank())
bp.append(flag_row(df["BP Systolic (mmHg)"] >= HTN_S, f"Systolic ≥{HTN_S} mmHg", denom=N))
bp.append(flag_row(df["BP Diastolic (mmHg)"] >= HTN_D, f"Diastolic ≥{HTN_D} mmHg", denom=N))
bp.append(flag_row(df["BP_high"], "Raised pressure on either measure", denom=N))
bp.append(flag_row(df["on_antihtn"], "Receiving antihypertensive treatment", denom=N))
bp.append(flag_row(df["htn"] == 1, "Hypertension (raised pressure or on treatment)", denom=N))
save_table(pd.DataFrame(bp), "t4_bmi_bp")

# ---------------------------------------------------------------- lifestyle
life = [flag_row(df["Smoker (Y/N)"].astype(str).str.startswith("Y"), "Current smoker", denom=N),
        flag_row(df["Drinker (Y/N)"].astype(str).str.startswith("Y"), "Consumes alcohol", denom=N),
        flag_row(df["Illicit Drug Use (Y/N)"].astype(str).str.startswith("Y"), "Illicit drug use", denom=N),
        flag_row(df["Herbal Use (Y/N)"].astype(str).str.startswith("Y"), "Uses herbal remedies", denom=N),
        blank()]
life += cat_rows(df["Exercise Frequency/week"], "Exercise, times per week",
                 ["Never", "Once", "Twice", "3 times", "More than 3 times"],
                 denom=int(df["Exercise Frequency/week"].notna().sum()))
save_table(pd.DataFrame(life), "t5_lifestyle")

n_sleep = int(df["Sleep"].notna().sum())
n_stress = int(df["Stress"].notna().sum())
RESULTS["n_sleep"], RESULTS["n_stress"] = n_sleep, n_stress
save_table(pd.DataFrame([cont_row("Sleep", "Sleep, hours per day"),
                         cont_row("Stress", "Perceived stress, 1 to 10")]), "t6_sleep_stress")

ss = cat_rows(pd.cut(df["Sleep"], [0, 6, 9, np.inf],
                     labels=["Fewer than 6 hours", "6 to 8 hours", "9 hours or more"], right=False),
              "Sleep duration", ["Fewer than 6 hours", "6 to 8 hours", "9 hours or more"], denom=n_sleep)
ss += cat_rows(pd.cut(df["Stress"], [0, 4, 7, np.inf], labels=["1 to 3", "4 to 6", "7 to 10"], right=False),
               "Perceived stress score", ["1 to 3", "4 to 6", "7 to 10"], denom=n_stress)
save_table(pd.DataFrame(ss), "t7_sleep_stress_cat")

# ---------------------------------------------------------------- chronic
chron = [flag_row(df["has_chronic"], "Any chronic condition reported", denom=N),
         flag_row(df["c_htn"], "    Hypertension", denom=N),
         flag_row(df["c_asthma"], "    Asthma", denom=N),
         flag_row(df["c_arthritis"], "    Arthritis", denom=N),
         flag_row(df["c_stroke"], "    Previous stroke", denom=N),
         flag_row(df["c_ulcer"], "    Peptic ulcer disease", denom=N),
         blank(),
         flag_row(df["on_antihtn"], "Receiving antihypertensive medication", denom=N)]
save_table(pd.DataFrame(chron), "t8_chronic")

n_vl = int(df["VL"].notna().sum())
RESULTS["n_vl"] = n_vl
save_table(pd.DataFrame([
    flag_row(df["VL_supp"] == 1, "Viral load below 50 copies/mL", denom=n_vl),
    flag_row(df["VL_supp"] == 0, "Viral load 50 copies/mL or above", denom=n_vl),
]), "t9_viral")

# ---------------------------------------------------------------- markers
save_table(pd.DataFrame([
    cont_row("HbA1c (%)", "HbA1c, %", dp=2),
    cont_row("GA_pct", "Glycated albumin, % of total albumin", dp=2),
    cont_row("Glyc. Albumin (GA) g/L", "Glycated albumin, g/L", dp=2),
    cont_row("Albumin (ALB) g/L", "Serum albumin, g/L", dp=1),
    cont_row("GA_HbA1c", "GA/HbA1c ratio", dp=2),
]), "t10_markers")

prev = [flag_row(df["HbA1c_cat"] == "Normoglycaemia", "Normoglycaemia (HbA1c <5.7%)", denom=N),
        flag_row(df["HbA1c_cat"] == "Prediabetes", "Prediabetes (HbA1c 5.7–6.4%)", denom=N),
        flag_row(df["HbA1c_cat"] == "Diabetes", "Diabetes (HbA1c ≥6.5%)", denom=N),
        flag_row(df["HbA1c_cat"] != "Normoglycaemia", "Hyperglycaemia (prediabetes or diabetes)", denom=N),
        blank(),
        flag_row(df["GA_pct"] >= GA_HYPER, f"Hyperglycaemia by glycated albumin (≥{GA_HYPER}%)", denom=N),
        flag_row(df["HbA1c (%)"] >= 6.0, "Hyperglycaemia at the locally derived HbA1c cut-off of 6.0%", denom=N)]
save_table(pd.DataFrame(prev), "t11_prevalence")
RESULTS["hyper_n"] = int((df["HbA1c_cat"] != "Normoglycaemia").sum())
RESULTS["ga_hyper_n"] = int((df["GA_pct"] >= GA_HYPER).sum())
RESULTS["ga_hyper_ci"] = wilson(RESULTS["ga_hyper_n"], N)
RESULTS["hb60_n"] = int((df["HbA1c (%)"] >= 6.0).sum())
RESULTS["hyper_ci"] = wilson(RESULTS["hyper_n"], N)

lo_ref, hi_ref = GA_REF
ga_ref = [flag_row(df["GA_pct"] < lo_ref, f"Below the reference interval (<{lo_ref}%)", denom=N),
          flag_row(df["GA_pct"].between(lo_ref, hi_ref),
                   f"Within the reference interval ({lo_ref}–{hi_ref}%)", denom=N),
          flag_row(df["GA_pct"] > hi_ref, f"Above the reference interval (>{hi_ref}%)", denom=N),
          blank(),
          flag_row(df["Albumin (ALB) g/L"] < 35, "Hypoalbuminaemia (albumin <35 g/L)", denom=N)]
save_table(pd.DataFrame(ga_ref), "t12_ga_ref")

RESULTS["alb_ga"] = stats.spearmanr(df["Albumin (ALB) g/L"], df["GA_pct"])


# ---------------------------------------------------------------- agreement
def passing_bablok(x, y, alpha=ALPHA):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    S = []
    for i in range(n - 1):
        dx, dy = x[i + 1:] - x[i], y[i + 1:] - y[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            S.append(np.where(dx != 0, dy / dx, np.nan))
    S = np.concatenate(S)
    S = np.sort(S[np.isfinite(S)])
    S = S[S != -1]
    Nn, K = len(S), int((S < -1).sum())

    def at(i1):
        return S[int(np.clip(i1, 1, Nn)) - 1]

    b = at((Nn + 1) / 2 + K) if Nn % 2 == 1 else 0.5 * (at(Nn / 2 + K) + at(Nn / 2 + 1 + K))
    a = float(np.median(y - b * x))
    C = stats.norm.ppf(1 - alpha / 2) * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    M1 = int(round((Nn - C) / 2))
    b_lo, b_hi = at(M1 + K), at(Nn - M1 + 1 + K)
    a_lo, a_hi = float(np.median(y - b_hi * x)), float(np.median(y - b_lo * x))
    return {"n": n, "slope": b, "slope_lo": b_lo, "slope_hi": b_hi,
            "intercept": a, "intercept_lo": a_lo, "intercept_hi": a_hi,
            "prop": not (b_lo <= 1 <= b_hi), "syst": not (a_lo <= 0 <= a_hi)}


def cusum(x, y, a, b):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = (y - (a + b * x))[np.argsort(x)]
    cs = np.cumsum(np.sign(r))
    st = np.max(np.abs(cs)) / np.sqrt(len(r))
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * st ** 2) for k in range(1, 100))
    return st, float(np.clip(p, 0, 1))


pb = passing_bablok(df["HbA1c (%)"], df["GA_pct"])
cs_stat, cs_p = cusum(df["HbA1c (%)"], df["GA_pct"], pb["intercept"], pb["slope"])
rho, rho_p = stats.spearmanr(df["HbA1c (%)"], df["GA_pct"])
pear, pear_p = stats.pearsonr(df["HbA1c (%)"], df["GA_pct"])
sub8 = df[df["HbA1c (%)"] < 8]
rho8, rho8_p = stats.spearmanr(sub8["HbA1c (%)"], sub8["GA_pct"])
pear8, pear8_p = stats.pearsonr(sub8["HbA1c (%)"], sub8["GA_pct"])
RESULTS.update(pb=pb, cusum=(cs_stat, cs_p), rho=(rho, rho_p), pear=(pear, pear_p),
               rho8=(rho8, rho8_p), pear8=(pear8, pear8_p), n8=len(sub8))

save_table(pd.DataFrame([
    {"Parameter": "Passing–Bablok slope", "Estimate": f"{pb['slope']:.3f}",
     "95% CI": f"{pb['slope_lo']:.3f} to {pb['slope_hi']:.3f}",
     "Interpretation": "Proportional difference" if pb["prop"] else "No proportional difference"},
    {"Parameter": "Passing–Bablok intercept", "Estimate": f"{pb['intercept']:.3f}",
     "95% CI": f"{pb['intercept_lo']:.3f} to {pb['intercept_hi']:.3f}",
     "Interpretation": "Systematic difference" if pb["syst"] else "No systematic difference"},
    {"Parameter": "CUSUM test of linearity", "Estimate": f"{cs_stat:.3f}", "95% CI": "—",
     "Interpretation": f"p = {fmt_p(cs_p)}; linearity rejected" if cs_p < ALPHA else f"p = {fmt_p(cs_p)}"},
    {"Parameter": "Spearman rho, all participants", "Estimate": f"{rho:.3f}", "95% CI": "—",
     "Interpretation": f"p = {fmt_p(rho_p)}; monotonic association"},
    {"Parameter": "Pearson r, all participants", "Estimate": f"{pear:.3f}", "95% CI": "—",
     "Interpretation": f"p = {fmt_p(pear_p)}; sensitive to leverage"},
    {"Parameter": f"Spearman rho, HbA1c below 8% (n = {len(sub8)})", "Estimate": f"{rho8:.3f}",
     "95% CI": "—", "Interpretation": f"p = {fmt_p(rho8_p)}"},
    {"Parameter": f"Pearson r, HbA1c below 8% (n = {len(sub8)})", "Estimate": f"{pear8:.3f}",
     "95% CI": "—", "Interpretation": f"p = {fmt_p(pear8_p)}"},
]), "t13_agreement")

GA_EQ = {t: pb["intercept"] + pb["slope"] * t for t in (ADA_PRE, ADA_DM)}
RESULTS["GA_EQ"] = GA_EQ

# --- categorical agreement against the external reference-interval upper limit
hb_hyper = df["HbA1c (%)"] >= HB_HYPER
ga_hyper = df["GA_pct"] >= GA_HYPER
ct = pd.crosstab(pd.Categorical(hb_hyper.map({True: "Hyperglycaemic", False: "Normoglycaemic"}),
                                ["Hyperglycaemic", "Normoglycaemic"]),
                 pd.Categorical(ga_hyper.map({True: f"GA ≥{GA_HYPER}%", False: f"GA <{GA_HYPER}%"}),
                                [f"GA <{GA_HYPER}%", f"GA ≥{GA_HYPER}%"]))
ct.index.name = "HbA1c"
save_table(ct.reset_index(), "t14_crosstab")
RESULTS["ct"] = ct

po = float((hb_hyper == ga_hyper).mean())
pe = hb_hyper.mean() * ga_hyper.mean() + (1 - hb_hyper.mean()) * (1 - ga_hyper.mean())
kappa = (po - pe) / (1 - pe)
se = np.sqrt(po * (1 - po) / (N * (1 - pe) ** 2))
z = stats.norm.ppf(1 - ALPHA / 2)
b_, c_ = int((~hb_hyper & ga_hyper).sum()), int((hb_hyper & ~ga_hyper).sum())
mc = (abs(b_ - c_) - 1) ** 2 / (b_ + c_) if (b_ + c_) else np.nan
mc_p = 1 - stats.chi2.cdf(mc, 1) if np.isfinite(mc) else np.nan
RESULTS.update(po=po, kappa=(kappa, kappa - z * se, kappa + z * se), mcnemar=(mc, mc_p, b_, c_))

save_table(pd.DataFrame([
    {"Statistic": "Observed agreement", "Value": f"{po*100:.1f}%"},
    {"Statistic": "Cohen's kappa (95% CI)", "Value": f"{kappa:.3f} ({kappa - z*se:.3f} to {kappa + z*se:.3f})"},
    {"Statistic": "McNemar chi-square (continuity corrected)", "Value": f"{mc:.3f}"},
    {"Statistic": "McNemar p", "Value": fmt_p(mc_p)},
    {"Statistic": "Discordant: hyperglycaemic by HbA1c only", "Value": str(c_)},
    {"Statistic": "Discordant: hyperglycaemic by glycated albumin only", "Value": str(b_)},
]), "t15_agreement_stats")

save_table(pd.DataFrame([
    flag_row(hb_hyper & ga_hyper, "Hyperglycaemic by both markers", denom=N),
    flag_row(hb_hyper & ~ga_hyper, "Hyperglycaemic by HbA1c only", denom=N),
    flag_row(~hb_hyper & ga_hyper, "Hyperglycaemic by glycated albumin only", denom=N),
    flag_row(~hb_hyper & ~ga_hyper, "Normoglycaemic by both markers", denom=N),
    flag_row(hb_hyper | ga_hyper, "Hyperglycaemic by either marker", denom=N),
]), "t16_added_detection")

# sensitivity of GA classification to the threshold used
thr_rows = []
for thr in [GA_HYPER_CI[0], GA_HYPER, GA_HYPER_CI[1], 15.2, 16.5]:
    gh = df["GA_pct"] >= thr
    po_t = float((hb_hyper == gh).mean())
    pe_t = hb_hyper.mean() * gh.mean() + (1 - hb_hyper.mean()) * (1 - gh.mean())
    k_t = (po_t - pe_t) / (1 - pe_t) if pe_t != 1 else np.nan
    thr_rows.append({"Glycated albumin threshold": f"≥{thr}%",
                     "Classified hyperglycaemic, n (%)": f"{int(gh.sum())} ({gh.mean()*100:.1f})",
                     "Observed agreement": f"{po_t*100:.1f}%",
                     "Cohen's kappa": f"{k_t:.3f}"})
save_table(pd.DataFrame(thr_rows), "t17_threshold_sensitivity")


# ---------------------------------------------------------------- logistic regression
def logit(y, X, names):
    X = np.column_stack([np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])])
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    llf = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    pb_ = y.mean()
    lln = float(np.sum(y * np.log(pb_) + (1 - y) * np.log(1 - pb_)))
    cov = np.linalg.inv(X.T @ (X * np.clip(p * (1 - p), 1e-10, None)[:, None]))
    sebeta = np.sqrt(np.diag(cov))
    zz = beta / sebeta
    pv = 2 * (1 - stats.norm.cdf(np.abs(zz)))
    zc = stats.norm.ppf(1 - ALPHA / 2)
    out = pd.DataFrame({"Term": ["const"] + names, "aOR": np.exp(beta).round(3),
                        "95% CI": [f"{np.exp(bb - zc*ss):.3f}–{np.exp(bb + zc*ss):.3f}"
                                   for bb, ss in zip(beta, sebeta)],
                        "p": [fmt_p(x) for x in pv]})
    return out[out["Term"] != "const"].reset_index(drop=True), 1 - llf / lln


def vif_table(X, names):
    Xc = np.column_stack([np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])])
    out = []
    for i in range(1, Xc.shape[1]):
        yv = Xc[:, i]
        others = np.delete(Xc, i, axis=1)
        bb, *_ = np.linalg.lstsq(others, yv, rcond=None)
        r = yv - others @ bb
        sst = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - (r ** 2).sum() / sst if sst > 0 else 0.0
        out.append({"Term": names[i - 1], "VIF": round(1 / (1 - r2) if r2 < 1 else np.inf, 2)})
    return pd.DataFrame(out)


models = []
vifs = []
for col, label in [("GA_pct", "Glycated albumin (per 1%)"),
                   ("HbA1c (%)", "HbA1c (per 1%)"),
                   ("GA_HbA1c", "GA/HbA1c ratio (per 1 unit)")]:
    d = df[["htn", col, "Age (yrs)", "Sex", "BMI"]].dropna()
    X = np.column_stack([d[col].values, d["Age (yrs)"].values,
                         d["Sex"].str.startswith("M").astype(int).values, d["BMI"].values])
    names = [label, "Age (per year)", "Male sex", "Body mass index (per 1 kg/m²)"]
    out, pr2 = logit(d["htn"].astype(int).values, X, names)
    out.insert(0, "Model", [label] + [""] * (len(out) - 1))
    models.append(out)
    vifs.append(vif_table(X, names).assign(Model=label))
    RESULTS.setdefault("models", {})[col] = (out, pr2, int(d["htn"].sum()), len(d))

model_tab = pd.concat(models, ignore_index=True)
save_table(model_tab, "t18_hypertension_models")
save_table(pd.concat(vifs, ignore_index=True)[["Model", "Term", "VIF"]], "t19_vif")

p_htn, lo_htn, hi_htn = wilson(int(df["htn"].sum()), N)
RESULTS["htn"] = (int(df["htn"].sum()), p_htn, lo_htn, hi_htn)

# ---------------------------------------------------------------- lipids
save_table(pd.DataFrame([
    cont_row("TC (mmol/L)", "Total cholesterol, mmol/L", dp=2),
    cont_row("HDL (mmol/L)", "HDL cholesterol, mmol/L", dp=2),
    cont_row("LDL (mmol/L)", "LDL cholesterol, mmol/L", dp=2),
    cont_row("TG (mmol/L)", "Triglycerides, mmol/L", dp=2),
    cont_row("nonHDL", "Non-HDL cholesterol, mmol/L", dp=2),
    cont_row("TC_HDL", "Total cholesterol to HDL ratio", dp=2),
    cont_row("AIP", "Atherogenic index of plasma", dp=3),
]), "t20_lipids")

dys = [flag_row(df["TC_high"], "Total cholesterol >5.2 mmol/L", denom=N),
       flag_row(df["LDL_high"], "LDL cholesterol >3.4 mmol/L", denom=N),
       flag_row(df["TG_high"], "Triglycerides >1.7 mmol/L", denom=N),
       flag_row(df["HDL_low"], "Low HDL cholesterol (sex-specific)", denom=N),
       flag_row(df["any_dys"], "At least one lipid abnormality", denom=N),
       blank()]
dys += [flag_row(df["AIP_band"] == b, f"Atherogenic index, {b.lower()} risk band", denom=N)
        for b in ["Low", "Medium", "High"]]
save_table(pd.DataFrame(dys), "t21_dyslipidaemia")

corr = []
for col, lab in [("HbA1c (%)", "HbA1c"), ("GA_pct", "Glycated albumin"),
                 ("GA_HbA1c", "GA/HbA1c ratio")]:
    for y, ylab in [("AIP", "Atherogenic index of plasma"), ("TG (mmol/L)", "Triglycerides"),
                    ("HDL (mmol/L)", "HDL cholesterol"), ("nonHDL", "Non-HDL cholesterol"),
                    ("BMI", "Body mass index"), ("Albumin (ALB) g/L", "Serum albumin")]:
        r, p = stats.spearmanr(df[col], df[y], nan_policy="omit")
        corr.append({"Marker": lab, "Against": ylab, "n": int((df[col].notna() & df[y].notna()).sum()),
                     "Spearman rho": round(r, 3), "p": fmt_p(p)})
corr_tab = pd.DataFrame(corr)
save_table(corr_tab, "t22_correlations")
RESULTS["corr"] = corr_tab

# ---------------------------------------------------------------- sensitivity
clean = df[~df["qc"]]
pbc = passing_bablok(clean["HbA1c (%)"], clean["GA_pct"])
rc, rcp = stats.spearmanr(clean["HbA1c (%)"], clean["GA_pct"])
sens_rows = []
for name, d, pbx, rr in [("All records", df, pb, (rho, rho_p)),
                         ("Flagged records excluded", clean, pbc, (rc, rcp))]:
    k = int(d["htn"].sum())
    p_, l_, h_ = wilson(k, len(d))
    sens_rows.append({
        "Dataset": name, "n": len(d),
        "Hyperglycaemia % (95% CI)": "{:.1f} ({:.1f}–{:.1f})".format(
            *wilson(int((d["HbA1c_cat"] != "Normoglycaemia").sum()), len(d))),
        "Hypertension % (95% CI)": f"{p_:.1f} ({l_:.1f}–{h_:.1f})",
        "Spearman rho (HbA1c vs GA)": f"{rr[0]:.3f} (p = {fmt_p(rr[1])})",
        "Passing–Bablok slope": f"{pbx['slope']:.3f}",
    })
save_table(pd.DataFrame(sens_rows), "t23_sensitivity")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(6.6, 3.3))
bins = np.arange(15, 85, 5); centres = bins[:-1] + 2.5; w = 2.0
for i, sx in enumerate(["Female", "Male"]):
    cnt, _ = np.histogram(df.loc[df["Sex"] == sx, "Age (yrs)"], bins=bins)
    ax.bar(centres + (i - 0.5) * w, cnt, width=w - 0.25, color=CAT[i],
           label=f"{sx} (n = {int((df['Sex'] == sx).sum())})", linewidth=0)
ax.set_xlabel("Age, years"); ax.set_ylabel("Participants")
ax.set_title("Age distribution by sex")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
save_fig(fig, "f1_age_sex")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
edu = df["Education Level"].value_counts().reindex(["Primary", "Secondary", "Tertiary"])
axes[0].barh(list(edu.index)[::-1], list(edu.values)[::-1], color=C_BLUE, height=0.6, linewidth=0)
for i, v in enumerate(list(edu.values)[::-1]):
    axes[0].text(v + 1.5, i, f"{v} ({v/N*100:.0f}%)", va="center", fontsize=9, color=MUTED)
axes[0].set_xlim(0, edu.max() * 1.3); axes[0].set_title("Highest level of education", fontsize=9.5)
axes[0].set_xlabel("Participants"); axes[0].grid(axis="y", visible=False)
order = ["Formal employment", "Self-employed or informal work", "Not in paid work", "Not recorded"]
emp = df["Employment"].value_counts().reindex(order).fillna(0).astype(int)
short = ["Formal", "Self-employed\nor informal", "Not in paid\nwork", "Not recorded"]
axes[1].barh(short[::-1], list(emp.values)[::-1], color=C_GREEN, height=0.6, linewidth=0)
for i, v in enumerate(list(emp.values)[::-1]):
    axes[1].text(v + 1.0, i, f"{v} ({v/N*100:.0f}%)", va="center", fontsize=9, color=MUTED)
axes[1].set_xlim(0, emp.max() * 1.4); axes[1].set_title("Employment status", fontsize=9.5)
axes[1].set_xlabel("Participants"); axes[1].grid(axis="y", visible=False)
fig.suptitle("Education and employment", x=0.005, ha="left",
             fontsize=10.5, fontweight="bold", y=1.04)
save_fig(fig, "f2_edu_emp")

fig, ax = plt.subplots(figsize=(6.8, 3.4))
cats = ["Underweight", "Normal weight", "Overweight", "Obese"]
xs = np.arange(len(cats)); wd = 0.38
for i, sx in enumerate(["Female", "Male"]):
    vals = [int(((df["Sex"] == sx) & (df["BMI_cat"] == c)).sum()) for c in cats]
    ax.bar(xs + (i - 0.5) * wd, vals, width=wd - 0.04, color=CAT[i],
           label=f"{sx} (n = {int((df['Sex'] == sx).sum())})", linewidth=0)
    for x, v in zip(xs + (i - 0.5) * wd, vals):
        if v:
            ax.text(x, v + 0.6, str(v), ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks(xs); ax.set_xticklabels(cats); ax.set_ylabel("Participants")
ax.set_title("Body mass index category by sex")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
save_fig(fig, "f3_bmi_sex")

fig, ax = plt.subplots(figsize=(6.2, 4.2))
h_ = df["htn"] == 1
ax.scatter(df.loc[~h_, "BP Systolic (mmHg)"], df.loc[~h_, "BP Diastolic (mmHg)"], s=34,
           color=C_BLUE, alpha=0.7, linewidth=1.0, edgecolor="white",
           label=f"Not hypertensive (n = {int((~h_).sum())})")
ax.scatter(df.loc[h_, "BP Systolic (mmHg)"], df.loc[h_, "BP Diastolic (mmHg)"], s=34,
           color=C_ORANGE, alpha=0.75, linewidth=1.0, edgecolor="white",
           label=f"Hypertensive (n = {int(h_.sum())})")
ax.axvline(HTN_S, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.axhline(HTN_D, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.set_xlabel("Systolic blood pressure, mmHg"); ax.set_ylabel("Diastolic blood pressure, mmHg")
ax.set_title("Blood pressure of the cohort")
ax.legend(loc="lower right")
save_fig(fig, "f4_bp")

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
specs = [(df["HbA1c (%)"], "HbA1c, %", C_BLUE, "ada"),
         (df["GA_pct"], "Glycated albumin, %", C_GREEN, "ref"),
         (df["GA_HbA1c"], "GA/HbA1c ratio", C_ORANGE, None)]
for ax, (s, lab, colour, mark) in zip(axes, specs):
    s = s.dropna()
    ax.hist(s, bins=22, color=colour, linewidth=0)
    if mark == "ada":
        for t in (ADA_PRE, ADA_DM):
            ax.axvline(t, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    if mark == "ref":
        ax.axvspan(*GA_REF, color=MUTED, alpha=0.12, zorder=0)
    ax.set_xlabel(f"{lab}\nn = {len(s)}")
    ax.set_ylabel("Participants" if ax is axes[0] else "")
    ax.grid(axis="x", visible=False)
fig.suptitle("Distribution of the glycaemic markers", x=0.005, ha="left",
             fontsize=10.5, fontweight="bold", y=1.05)
save_fig(fig, "f5_markers")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for i, cat in enumerate(CATS):
    m = df["HbA1c_cat"] == cat
    ax.scatter(df.loc[m, "HbA1c (%)"], df.loc[m, "GA_pct"], s=38, color=CAT[i], alpha=0.75,
               linewidth=1.0, edgecolor="white", label=f"{cat} (n = {int(m.sum())})")
ax.axhspan(*GA_REF, color=MUTED, alpha=0.10, zorder=0)
for t in (ADA_PRE, ADA_DM):
    ax.axvline(t, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
ax.set_xlabel(f"HbA1c, %\nAll {N}: Spearman ρ = {rho:.3f}, Pearson r = {pear:.3f}\n"
              f"HbA1c below 8% (n = {len(sub8)}): ρ = {rho8:.3f}, r = {pear8:.3f}")
ax.set_ylabel("Glycated albumin, %")
ax.set_title("Glycated albumin against HbA1c")
ax.legend(loc="upper left")
save_fig(fig, "f6_scatter")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
x, y = df["HbA1c (%)"], df["GA_pct"]
zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
mz, dz = (zx + zy) / 2, zy - zx
md_, sd_ = dz.mean(), dz.std()
ax.scatter(mz, dz, s=30, color=C_GREEN, alpha=0.65, linewidth=1.0, edgecolor="white",
           label="Standardised pairs")
ax.axhline(md_, color=C_ORANGE, label=f"Mean difference {md_:.3f}")
ax.axhline(md_ + 1.96 * sd_, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
           label="95% limits of agreement")
ax.axhline(md_ - 1.96 * sd_, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.set_xlabel("Mean of standardised markers (z)")
ax.set_ylabel("Glycated albumin z − HbA1c z")
ax.set_title("Difference plot on standardised markers")
ax.legend(loc="upper right", fontsize=8.5)
save_fig(fig, "f7_difference")
RESULTS["loa"] = (md_, md_ - 1.96 * sd_, md_ + 1.96 * sd_)

fig, ax = plt.subplots(figsize=(5.6, 4.0))
M = ct.values
ax.imshow(M, cmap="Blues", vmin=0)
ax.set_xticks(range(2)); ax.set_yticks(range(2))
ax.set_xticklabels(list(ct.columns)); ax.set_yticklabels(list(ct.index))
ax.set_xlabel("Glycated albumin"); ax.set_ylabel("HbA1c")
ax.set_title("Cross-classification of the two markers")
for i in range(2):
    for j in range(2):
        ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=14,
                color="white" if M[i, j] > M.max() * 0.55 else INK, fontweight="bold")
ax.grid(False)
for s_ in ax.spines.values():
    s_.set_visible(False)
save_fig(fig, "f8_crosstab")

fig, ax = plt.subplots(figsize=(6.8, 3.4))
series = [("Glycated albumin (per 1%)", RESULTS["models"]["GA_pct"][0].iloc[0], C_BLUE),
          ("HbA1c (per 1%)", RESULTS["models"]["HbA1c (%)"][0].iloc[0], C_ORANGE),
          ("GA/HbA1c ratio (per 1 unit)", RESULTS["models"]["GA_HbA1c"][0].iloc[0], C_GREEN)]
ys = np.arange(len(series))[::-1]
for (lab, row, colour), yi in zip(series, ys):
    lo_, hi_ = [float(v) for v in row["95% CI"].split("–")]
    est = float(row["aOR"])
    ax.errorbar(est, yi, xerr=[[est - lo_], [hi_ - est]], fmt="o", color=colour, ecolor=colour,
                elinewidth=1.8, capsize=3, markersize=8, markeredgecolor="white", markeredgewidth=1.2)
ax.axvline(1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.set_yticks(ys); ax.set_yticklabels([s_[0] for s_ in series])
ax.set_xscale("log"); ax.set_xlabel("Adjusted odds ratio for hypertension (log scale)")
ax.set_title("Each glycaemic marker against hypertension")
ax.grid(axis="y", visible=False)
x0, x1 = ax.get_xlim(); ax.set_xlim(x0, x1 * 2.2)
for (lab, row, colour), yi in zip(series, ys):
    ax.text(x1 * 1.08, yi, f"p = {row['p']}", va="center", fontsize=8.5, color=MUTED)
save_fig(fig, "f9_forest")

fracs = [("TC (mmol/L)", "Total cholesterol", LIPID["TC"]),
         ("LDL (mmol/L)", "LDL cholesterol", LIPID["LDL"]),
         ("HDL (mmol/L)", "HDL cholesterol", None),
         ("TG (mmol/L)", "Triglycerides", LIPID["TG"])]
fig, axes = plt.subplots(1, 4, figsize=(9.8, 3.2))
for ax, (col, lab, cut) in zip(axes, fracs):
    s = df[col].dropna()
    ax.boxplot(s, widths=0.45, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, linewidth=1.6),
               boxprops=dict(facecolor=C_BLUE, alpha=0.25, edgecolor=C_BLUE, linewidth=1.4),
               whiskerprops=dict(color=C_BLUE, linewidth=1.2),
               capprops=dict(color=C_BLUE, linewidth=1.2))
    ax.scatter(np.random.normal(1, 0.055, len(s)), s, s=11, color=C_BLUE, alpha=0.45, linewidth=0)
    if cut:
        ax.axhline(cut, color=C_ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.text(1.38, cut, f" {cut}", color=C_ORANGE, va="center", fontsize=8.5)
    ax.set_title(lab, fontsize=9.5); ax.set_xticks([])
    ax.set_ylabel("mmol/L" if ax is axes[0] else ""); ax.grid(axis="x", visible=False)
fig.suptitle("Distribution of lipid fractions with reference cut-points",
             x=0.005, ha="left", fontsize=10.5, fontweight="bold", y=1.05)
save_fig(fig, "f10_lipids")

fig, ax = plt.subplots(figsize=(6.6, 3.4))
s = df["AIP"].dropna()
ax.hist(s, bins=24, color=C_BLUE, linewidth=0)
ymax = ax.get_ylim()[1]; ax.set_ylim(0, ymax * 1.18)
for xv in (0.10, 0.24):
    ax.vlines(xv, 0, ymax * 1.02, color=C_ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
xlo, xhi = ax.get_xlim()
cnt = df["AIP_band"].value_counts().reindex(["Low", "Medium", "High"])
for lab, xp in [("Low", (xlo + 0.10) / 2), ("Medium", 0.17), ("High", (0.24 + xhi) / 2)]:
    nv = int(cnt[lab])
    ax.text(xp, ymax * 1.10, f"{lab}\n{nv} ({nv/N*100:.0f}%)", color=C_ORANGE,
            fontsize=8.5, ha="center", va="center", linespacing=1.3)
ax.set_xlabel("Atherogenic index of plasma, log₁₀(TG/HDL-C)"); ax.set_ylabel("Participants")
ax.set_title("Atherogenic index of plasma")
ax.grid(axis="x", visible=False)
save_fig(fig, "f11_aip")

# ---------------------------------------------------------------- export
with pd.ExcelWriter(OUT / "chapter3_tables.xlsx", engine="openpyxl") as xw:
    for f in sorted(glob.glob(str(TAB / "*.csv"))):
        pd.read_csv(f).to_excel(xw, sheet_name=Path(f).stem[:31], index=False)

# ---------------------------------------------------------------- report
print("=" * 70)
print(f"N = {N}")
print(f"hyperglycaemia (HbA1c >= 5.7%): {RESULTS['hyper_n']} "
      f"({RESULTS['hyper_ci'][0]:.1f}%, 95% CI {RESULTS['hyper_ci'][1]:.1f}-{RESULTS['hyper_ci'][2]:.1f})")
print(df["HbA1c_cat"].value_counts().reindex(CATS).to_string())
print(f"GA-defined hyperglycaemia (>={GA_HYPER}%): {RESULTS['ga_hyper_n']} "
      f"({RESULTS['ga_hyper_ci'][0]:.1f}%, {RESULTS['ga_hyper_ci'][1]:.1f}-{RESULTS['ga_hyper_ci'][2]:.1f})")
print(f"HbA1c >= 6.0% (local cut-off): {RESULTS['hb60_n']} ({RESULTS['hb60_n']/N*100:.1f}%)")
print(f"\nPassing-Bablok slope {pb['slope']:.3f} ({pb['slope_lo']:.3f} to {pb['slope_hi']:.3f}), "
      f"intercept {pb['intercept']:.3f} ({pb['intercept_lo']:.3f} to {pb['intercept_hi']:.3f})")
print(f"CUSUM {cs_stat:.3f}, p = {fmt_p(cs_p)}")
print(f"Spearman {rho:.3f} (p={fmt_p(rho_p)})  Pearson {pear:.3f} (p={fmt_p(pear_p)})")
print(f"HbA1c<8 (n={len(sub8)}): Spearman {rho8:.3f} (p={fmt_p(rho8_p)}), Pearson {pear8:.3f} (p={fmt_p(pear8_p)})")
print(f"GA equivalents: 5.7% -> {GA_EQ[ADA_PRE]:.2f}%, 6.5% -> {GA_EQ[ADA_DM]:.2f}%")
print(f"\nagreement {po*100:.1f}%, kappa {kappa:.3f} ({kappa-z*se:.3f} to {kappa+z*se:.3f})")
print(f"McNemar {mc:.3f}, p={fmt_p(mc_p)}; HbA1c-only {c_}, GA-only {b_}")
print(ct.to_string())
print(f"\nhypertension {RESULTS['htn'][0]}/{N} = {RESULTS['htn'][1]:.1f}% "
      f"({RESULTS['htn'][2]:.1f}-{RESULTS['htn'][3]:.1f})")
for k_, (out, pr2, ev, nn) in RESULTS["models"].items():
    print(f"\n-- {k_}: n={nn}, events={ev}, pseudo R2={pr2:.3f}")
    print(out.to_string(index=False))
print("\n" + pd.read_csv(TAB / "t20_lipids.csv").to_string(index=False))
print("\n" + pd.read_csv(TAB / "t21_dyslipidaemia.csv").to_string(index=False))
print("\n" + corr_tab.to_string(index=False))
print("\n" + pd.read_csv(TAB / "t17_threshold_sensitivity.csv").to_string(index=False))
print("\n" + pd.read_csv(TAB / "t23_sensitivity.csv").to_string(index=False))
print(f"\nalbumin vs GA%: rho={RESULTS['alb_ga'][0]:.3f}, p={fmt_p(RESULTS['alb_ga'][1])}")
print(f"LoA (z units): {RESULTS['loa'][0]:.3f} ({RESULTS['loa'][1]:.3f} to {RESULTS['loa'][2]:.3f})")
print(f"QC flagged: {RESULTS['qc_n']}")
print(f"sleep n={n_sleep}, stress n={n_stress}, VL n={n_vl}")
print("tables:", len(glob.glob(str(TAB / '*.csv'))), " figures:", len(glob.glob(str(FIG / '*.png'))))
