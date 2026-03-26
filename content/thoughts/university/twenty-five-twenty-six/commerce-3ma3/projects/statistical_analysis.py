#!/usr/bin/env python3
"""
Mister Maki Marketing Research: Statistical Analysis
Commerce 3MA3, McMaster University
---
Run: python statistical_analysis.py
Output: /tmp/mister_maki_charts/analysis_results.txt + chart PNGs
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    mannwhitneyu,
    pointbiserialr,
    friedmanchisquare,
    spearmanr,
    pearsonr,
    kruskal,
    shapiro,
    norm,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

OUTDIR = Path('/tmp/mister_maki_charts')
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTDIR / 'analysis_results.txt'

DATA_PATH = Path(__file__).parent / 'MisterMaki_Survey_Response.xlsx'

BG = '#FFFDF7'
C1 = '#ea9a97'
C2 = '#a692c3'
C3 = '#6e8b74'
C4 = '#d4a373'
C5 = '#7fa5b5'
PALETTE = [C1, C2, C3, C4, C5, '#c9ada7', '#9a8c98', '#f2cc8f']
TEXT_COLOR = '#2b2b2b'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

results = []

def log(text: str):
    results.append(text)
    print(text)

def section(title: str):
    bar = '=' * 72
    log(f"\n{bar}")
    log(f"  {title.upper()}")
    log(bar)

def subsection(title: str):
    log(f"\n--- {title} ---")


# ============================================================
#  COLUMN MAPPING
# ============================================================
# The survey is from Google Forms (xlsx export).
# Column headers are full question text.
# We match columns by searching for keywords.
# If a mapping fails, the inspection output (Section 0) shows
# all columns so you can fix the keywords below.

KEYWORD_MAP = {
    'timestamp':        ['timestamp', 'time'],
    'student_status':   ['mcmaster', 'student', 'affiliation', 'are you a'],
    'age':              ['age', 'old'],
    'gender':           ['gender', 'sex', 'identify'],
    'awareness':        ['heard of mister maki', 'aware', 'know about mister maki', 'heard of'],
    'discovery':        ['how did you', 'hear about', 'discover', 'first learn', 'find out'],
    'trial':            ['ordered from', 'tried', 'purchased', 'eaten at', 'visited mister maki', 'have you ever'],
    'temaki_familiar':  ['familiar', 'temaki', 'hand roll', 'handroll'],
    'weekly_budget':    ['budget', 'spend', 'weekly', 'food budget', 'how much'],
    'dining_freq':      ['how often', 'dining', 'eat out', 'frequency', 'times per week'],
    'value_perception': ['value', 'worth', 'price', 'affordable', 'money'],
    'reorder_intent':   ['reorder', 'order again', 'return', 'come back', 'revisit', 'likely to order'],
    'overall_sat':      ['overall', 'satisfaction', 'experience'],
    'sat_food':         ['food quality', 'taste', 'freshness'],
    'sat_price':        ['price', 'pricing', 'cost', 'affordab'],
    'sat_portion':      ['portion', 'size', 'amount'],
    'sat_convenience':  ['convenience', 'location', 'access', 'convenient'],
    'sat_service':      ['service', 'staff', 'friendly'],
    'comp_pitapit':     ['pita pit'],
    'comp_burrito':     ['burrito bandidos', 'burrito'],
    'comp_saigon':      ['saigon', 'asian'],
    'recommend':        ['recommend', 'nps', 'tell a friend', 'suggest'],
    'cuisine_pref':     ['cuisine', 'type of food', 'prefer'],
}


def find_column(df: pd.DataFrame, key: str, keywords: list[str]) -> Optional[str]:
    cols_lower = {c: c.lower() for c in df.columns}
    for kw in keywords:
        for orig, low in cols_lower.items():
            if kw.lower() in low:
                return orig
    return None


def map_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    mapping = {}
    used = set()
    for key, keywords in KEYWORD_MAP.items():
        col = find_column(df, key, keywords)
        if col and col not in used:
            mapping[key] = col
            used.add(col)
        else:
            mapping[key] = None
    sat_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(s in cl for s in ['satisfaction', 'rate', 'rating']) and c not in used:
            is_sat = any(dim in cl for dim in ['food', 'quality', 'taste', 'price',
                         'portion', 'size', 'convenience', 'location', 'service',
                         'staff', 'overall', 'experience', 'freshness', 'speed',
                         'atmosphere', 'cleanliness', 'menu', 'variety'])
            if is_sat:
                sat_cols.append(c)
    mapping['_sat_cols'] = sat_cols if sat_cols else None
    comp_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(s in cl for s in ['pita', 'burrito', 'saigon', 'competitor', 'compare']):
            comp_cols.append(c)
    mapping['_comp_cols'] = comp_cols if comp_cols else None
    unmapped = [c for c in df.columns if c not in used and c not in (sat_cols + comp_cols)]
    mapping['_unmapped'] = unmapped
    return mapping


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, centre - spread), min(1, centre + spread))


def cramers_v(table: np.ndarray) -> float:
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    min_dim = min(table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1 - (2 * u) / (n1 * n2)


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1)**2 + (n2 - 1) * g2.std(ddof=1)**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float('nan')
    return (g1.mean() - g2.mean()) / pooled_std


def interpret_effect_chi(v: float) -> str:
    if v < 0.1: return 'negligible'
    if v < 0.3: return 'small'
    if v < 0.5: return 'medium'
    return 'large'


def interpret_effect_d(d: float) -> str:
    d = abs(d)
    if d < 0.2: return 'negligible'
    if d < 0.5: return 'small'
    if d < 0.8: return 'medium'
    return 'large'


def safe_chi2_or_fisher(table: np.ndarray, label: str):
    expected = chi2_contingency(table, correction=False)[3]
    low_expected = (expected < 5).sum()
    total_cells = expected.size

    if table.shape == (2, 2) and (low_expected > 0 or table.sum() < 40):
        odds, p = fisher_exact(table)
        v = cramers_v(table)
        log(f"  Test: Fisher's Exact (2x2, {low_expected}/{total_cells} cells with E<5)")
        log(f"  Odds Ratio = {odds:.3f}, p = {p:.4f}")
        log(f"  Cramer's V = {v:.3f} ({interpret_effect_chi(v)})")
        return {'test': 'fisher', 'p': p, 'odds_ratio': odds, 'cramers_v': v}
    elif low_expected / total_cells > 0.2:
        if table.shape == (2, 2):
            odds, p = fisher_exact(table)
            v = cramers_v(table)
            log(f"  Test: Fisher's Exact (>20% cells with E<5)")
            log(f"  Odds Ratio = {odds:.3f}, p = {p:.4f}")
            log(f"  Cramer's V = {v:.3f} ({interpret_effect_chi(v)})")
            return {'test': 'fisher', 'p': p, 'odds_ratio': odds, 'cramers_v': v}
        else:
            chi2, p, dof, exp = chi2_contingency(table, correction=False)
            v = cramers_v(table)
            log(f"  WARNING: {low_expected}/{total_cells} cells have E<5. Results unreliable.")
            log(f"  Test: Chi-square (with caveat)")
            log(f"  chi2({dof}) = {chi2:.3f}, p = {p:.4f}")
            log(f"  Cramer's V = {v:.3f} ({interpret_effect_chi(v)})")
            return {'test': 'chi2_caveat', 'p': p, 'chi2': chi2, 'dof': dof, 'cramers_v': v}
    else:
        use_yates = table.shape == (2, 2)
        chi2, p, dof, exp = chi2_contingency(table, correction=use_yates)
        v = cramers_v(table)
        correction_note = " (Yates correction)" if use_yates else ""
        log(f"  Test: Chi-square{correction_note}")
        log(f"  chi2({dof}) = {chi2:.3f}, p = {p:.4f}")
        log(f"  Cramer's V = {v:.3f} ({interpret_effect_chi(v)})")
        return {'test': 'chi2', 'p': p, 'chi2': chi2, 'dof': dof, 'cramers_v': v}


# ============================================================
#  MAIN ANALYSIS
# ============================================================

def run_analysis():
    section("0. DATA INSPECTION")
    df = pd.read_excel(DATA_PATH)
    log(f"File: {DATA_PATH}")
    log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"")

    for i, col in enumerate(df.columns):
        nuniq = df[col].nunique()
        nulls = df[col].isnull().sum()
        dtype = df[col].dtype
        log(f"  [{i:2d}] {col}")
        log(f"       dtype={dtype}, unique={nuniq}, missing={nulls}")
        if nuniq <= 20:
            vc = df[col].value_counts(dropna=False)
            for val, cnt in vc.items():
                log(f"         {val}: {cnt}")
        else:
            log(f"         (top 5) {df[col].value_counts().head(5).to_dict()}")

    cmap = map_columns(df)
    section("1. COLUMN MAPPING")
    for key, col in cmap.items():
        if key.startswith('_'):
            continue
        status = col if col else '*** NOT FOUND ***'
        log(f"  {key:20s} -> {status}")
    if cmap.get('_sat_cols'):
        log(f"\n  Satisfaction columns detected: {len(cmap['_sat_cols'])}")
        for c in cmap['_sat_cols']:
            log(f"    - {c}")
    if cmap.get('_comp_cols'):
        log(f"\n  Competitor columns detected: {len(cmap['_comp_cols'])}")
        for c in cmap['_comp_cols']:
            log(f"    - {c}")
    if cmap.get('_unmapped'):
        log(f"\n  Unmapped columns ({len(cmap['_unmapped'])}):")
        for c in cmap['_unmapped']:
            log(f"    - {c}")

    n = len(df)

    section("2. MISSING DATA ANALYSIS")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    log(f"  Total missing values: {total_missing} / {n * df.shape[1]} ({100*total_missing/(n*df.shape[1]):.1f}%)")
    if total_missing > 0:
        log(f"  Columns with missing data:")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                log(f"    {col}: {df[col].isnull().sum()} ({100*df[col].isnull().sum()/n:.1f}%)")
    log(f"  Complete cases: {df.dropna().shape[0]} / {n} ({100*df.dropna().shape[0]/n:.1f}%)")

    section("3. DESCRIPTIVE STATISTICS")

    def describe_col(key, label):
        col = cmap.get(key)
        if col is None:
            log(f"\n  {label}: column not found, skipping")
            return None
        subsection(label)
        series = df[col].dropna()
        if series.dtype in ['int64', 'float64'] and series.nunique() > 5:
            log(f"  n={len(series)}, mean={series.mean():.2f}, median={series.median():.2f}")
            log(f"  std={series.std():.2f}, min={series.min()}, max={series.max()}")
            log(f"  skewness={series.skew():.2f}, kurtosis={series.kurtosis():.2f}")
            if len(series) >= 8:
                w, p_norm = shapiro(series)
                log(f"  Shapiro-Wilk: W={w:.3f}, p={p_norm:.4f} {'(normal)' if p_norm > 0.05 else '(non-normal)'}")
        else:
            vc = series.value_counts()
            for val, cnt in vc.items():
                pct = 100 * cnt / len(series)
                log(f"  {val}: {cnt} ({pct:.1f}%)")
        return series

    describe_col('student_status', 'Student Status')
    describe_col('awareness', 'Brand Awareness')
    describe_col('trial', 'Trial / Purchase')
    describe_col('temaki_familiar', 'Temaki Familiarity')
    describe_col('weekly_budget', 'Weekly Food Budget')
    describe_col('dining_freq', 'Dining Frequency')
    describe_col('value_perception', 'Value Perception')
    describe_col('discovery', 'Discovery Channel')
    describe_col('reorder_intent', 'Reorder Intent')

    if cmap.get('_sat_cols'):
        subsection('Satisfaction Dimensions')
        for col in cmap['_sat_cols']:
            s = df[col].dropna()
            if len(s) > 0:
                if s.dtype in ['int64', 'float64']:
                    log(f"  {col[:60]:60s}  n={len(s):2d}  mean={s.mean():.2f}  std={s.std():.2f}")
                else:
                    log(f"  {col[:60]:60s}  n={len(s):2d}  mode={s.mode().iloc[0] if len(s.mode()) > 0 else 'N/A'}")

    # ============================================================
    section("4. HYPOTHESIS TESTS")
    # ============================================================

    alpha = 0.05
    test_results = {}

    # --- H1: Brand awareness x student status ---
    subsection("H1: Brand awareness higher among McMaster students than non-students")
    log("  H0: Awareness rate is equal across student/non-student groups")
    log("  H1: Awareness rate is higher among McMaster students")
    c_aware = cmap.get('awareness')
    c_student = cmap.get('student_status')
    if c_aware and c_student:
        df_h1 = df[[c_student, c_aware]].dropna()
        student_vals = df_h1[c_student].unique()
        log(f"  Student status values: {student_vals.tolist()}")

        def is_student(val):
            s = str(val).lower()
            return any(k in s for k in ['yes', 'mcmaster', 'student', 'true', '1'])

        def is_positive(val):
            s = str(val).lower()
            return any(k in s for k in ['yes', 'true', '1', 'heard', 'aware', 'know'])

        df_h1['_student'] = df_h1[c_student].apply(is_student)
        df_h1['_aware'] = df_h1[c_aware].apply(is_positive)

        ct = pd.crosstab(df_h1['_student'], df_h1['_aware'])
        log(f"\n  Contingency table (student x aware):")
        log(f"  {ct.to_string()}")

        n_students = df_h1['_student'].sum()
        n_nonstudents = (~df_h1['_student']).sum()
        log(f"\n  Students: n={n_students}, Non-students: n={n_nonstudents}")

        if ct.shape == (2, 2) or (ct.shape[0] >= 2 and ct.shape[1] >= 2):
            result = safe_chi2_or_fisher(ct.values, 'H1')
            test_results['H1'] = result

            if n_students > 0 and n_nonstudents > 0:
                aware_student = df_h1[df_h1['_student']]['_aware'].mean()
                aware_nonstudent = df_h1[~df_h1['_student']]['_aware'].mean()
                ci_s = wilson_ci(aware_student, n_students)
                ci_ns = wilson_ci(aware_nonstudent, n_nonstudents)
                log(f"\n  Awareness rate (students):     {aware_student:.1%}  95% CI [{ci_s[0]:.1%}, {ci_s[1]:.1%}]")
                log(f"  Awareness rate (non-students): {aware_nonstudent:.1%}  95% CI [{ci_ns[0]:.1%}, {ci_ns[1]:.1%}]")
                diff = aware_student - aware_nonstudent
                log(f"  Difference: {diff:+.1%}")
        else:
            log("  Cannot form 2x2 table (possibly all respondents in one group)")
            log(f"  Crosstab shape: {ct.shape}")
            test_results['H1'] = {'test': 'skipped', 'reason': 'insufficient groups'}
    else:
        log("  Required columns not found. Skipping.")
        test_results['H1'] = {'test': 'skipped', 'reason': 'missing columns'}

    # --- H2: Discovery channel x student status ---
    subsection("H2: Students more likely to discover via social media/word-of-mouth")
    log("  H0: Discovery channel distribution is independent of student status")
    log("  H1: Students over-index on social media and word-of-mouth")
    c_disc = cmap.get('discovery')
    if c_disc and c_student:
        df_h2 = df[[c_student, c_disc]].dropna()
        df_h2['_student'] = df_h2[c_student].apply(is_student)

        log(f"\n  Discovery channel values:")
        for val, cnt in df_h2[c_disc].value_counts().items():
            log(f"    {val}: {cnt}")

        def is_social_wom(val):
            s = str(val).lower()
            return any(k in s for k in ['social', 'instagram', 'tiktok', 'facebook',
                       'twitter', 'word', 'mouth', 'friend', 'peer', 'wom'])

        df_h2['_social_wom'] = df_h2[c_disc].apply(is_social_wom)
        ct2 = pd.crosstab(df_h2['_student'], df_h2['_social_wom'])
        log(f"\n  Contingency table (student x social/WOM discovery):")
        log(f"  {ct2.to_string()}")

        if ct2.shape[0] >= 2 and ct2.shape[1] >= 2:
            result = safe_chi2_or_fisher(ct2.values, 'H2')
            test_results['H2'] = result

            social_student = df_h2[df_h2['_student']]['_social_wom'].mean()
            social_nonstudent = df_h2[~df_h2['_student']]['_social_wom'].mean()
            n_s = df_h2['_student'].sum()
            n_ns = (~df_h2['_student']).sum()
            if n_s > 0:
                ci_s = wilson_ci(social_student, n_s)
                log(f"  Social/WOM rate (students):     {social_student:.1%}  95% CI [{ci_s[0]:.1%}, {ci_s[1]:.1%}]")
            if n_ns > 0:
                ci_ns = wilson_ci(social_nonstudent, n_ns)
                log(f"  Social/WOM rate (non-students): {social_nonstudent:.1%}  95% CI [{ci_ns[0]:.1%}, {ci_ns[1]:.1%}]")
        else:
            log("  Cannot form adequate contingency table")
            test_results['H2'] = {'test': 'skipped', 'reason': 'insufficient variation'}
    else:
        log("  Required columns not found. Skipping.")
        test_results['H2'] = {'test': 'skipped', 'reason': 'missing columns'}

    # --- H3: Perceived value x student status ---
    subsection("H3: Perceived value lower among Hamilton residents than McMaster students")
    log("  H0: Value perception does not differ between students and non-students")
    log("  H1: Non-students (Hamilton residents) report lower perceived value")
    c_value = cmap.get('value_perception')
    if c_value and c_student:
        df_h3 = df[[c_student, c_value]].dropna()
        df_h3['_student'] = df_h3[c_student].apply(is_student)

        def to_numeric_ordinal(series):
            if series.dtype in ['int64', 'float64']:
                return series
            ordinal_map = {}
            for val in series.unique():
                s = str(val).lower()
                if any(k in s for k in ['very low', 'strongly disagree', 'very poor', '1']):
                    ordinal_map[val] = 1
                elif any(k in s for k in ['low', 'disagree', 'poor', 'below', '2']):
                    ordinal_map[val] = 2
                elif any(k in s for k in ['neutral', 'moderate', 'average', 'neither', '3']):
                    ordinal_map[val] = 3
                elif any(k in s for k in ['high', 'agree', 'good', 'above', '4']):
                    ordinal_map[val] = 4
                elif any(k in s for k in ['very high', 'strongly agree', 'excellent', 'very good', '5']):
                    ordinal_map[val] = 5
                else:
                    try:
                        ordinal_map[val] = float(val)
                    except (ValueError, TypeError):
                        ordinal_map[val] = np.nan
            return series.map(ordinal_map)

        df_h3['_value_num'] = to_numeric_ordinal(df_h3[c_value])
        df_h3 = df_h3.dropna(subset=['_value_num'])

        students = df_h3[df_h3['_student']]['_value_num']
        nonstudents = df_h3[~df_h3['_student']]['_value_num']

        log(f"\n  Students:     n={len(students)}, mean={students.mean():.2f}, median={students.median():.1f}, std={students.std():.2f}")
        log(f"  Non-students: n={len(nonstudents)}, mean={nonstudents.mean():.2f}, median={nonstudents.median():.1f}, std={nonstudents.std():.2f}")

        if len(students) >= 3 and len(nonstudents) >= 3:
            u_stat, p_mwu = mannwhitneyu(students, nonstudents, alternative='two-sided')
            r_rb = rank_biserial(u_stat, len(students), len(nonstudents))
            d = cohens_d(students.values, nonstudents.values)
            log(f"\n  Mann-Whitney U = {u_stat:.1f}, p = {p_mwu:.4f}")
            log(f"  Rank-biserial r = {r_rb:.3f}")
            log(f"  Cohen's d = {d:.3f} ({interpret_effect_d(d)})")
            test_results['H3'] = {'test': 'mann_whitney', 'p': p_mwu, 'U': u_stat,
                                  'rank_biserial': r_rb, 'cohens_d': d}
        elif len(students) >= 1 and len(nonstudents) >= 1:
            log("  Sample too small for Mann-Whitney U (need n>=3 per group)")
            log(f"  Descriptive comparison only: student mean - nonstudent mean = {students.mean() - nonstudents.mean():+.2f}")
            test_results['H3'] = {'test': 'descriptive_only', 'reason': 'n too small'}
        else:
            log("  One group has no data")
            test_results['H3'] = {'test': 'skipped', 'reason': 'empty group'}
    else:
        log("  Required columns not found. Skipping.")
        test_results['H3'] = {'test': 'skipped', 'reason': 'missing columns'}

    # --- H4: Temaki familiarity x trial ---
    subsection("H4: Temaki familiarity correlated with having ordered from Mister Maki")
    log("  H0: Temaki familiarity and trial are independent")
    log("  H1: Higher temaki familiarity is associated with higher trial rate")
    c_fam = cmap.get('temaki_familiar')
    c_trial = cmap.get('trial')
    if c_fam and c_trial:
        df_h4 = df[[c_fam, c_trial]].dropna()

        def to_binary(series):
            if series.dtype in ['int64', 'float64']:
                return (series > 0).astype(int)
            return series.apply(lambda v: 1 if any(k in str(v).lower() for k in ['yes', 'true', '1']) else 0)

        df_h4['_trial_bin'] = to_binary(df_h4[c_trial])
        df_h4['_fam_num'] = to_numeric_ordinal(df_h4[c_fam])
        df_h4 = df_h4.dropna(subset=['_fam_num'])

        if df_h4['_fam_num'].nunique() > 2 and df_h4['_trial_bin'].nunique() == 2:
            r_pb, p_pb = pointbiserialr(df_h4['_trial_bin'], df_h4['_fam_num'])
            log(f"\n  Point-biserial correlation: r = {r_pb:.3f}, p = {p_pb:.4f}")
            log(f"  Effect: {'small' if abs(r_pb) < 0.3 else 'medium' if abs(r_pb) < 0.5 else 'large'}")
            test_results['H4'] = {'test': 'point_biserial', 'p': p_pb, 'r': r_pb}
        else:
            ct4 = pd.crosstab(df_h4[c_fam], df_h4['_trial_bin'])
            log(f"\n  Contingency table:")
            log(f"  {ct4.to_string()}")
            if ct4.shape[0] >= 2 and ct4.shape[1] >= 2:
                result = safe_chi2_or_fisher(ct4.values, 'H4')
                test_results['H4'] = result
            else:
                log("  Insufficient variation for test")
                test_results['H4'] = {'test': 'skipped'}

        rho, p_rho = spearmanr(df_h4['_fam_num'], df_h4['_trial_bin'])
        log(f"  Spearman's rho = {rho:.3f}, p = {p_rho:.4f} (ordinal correlation check)")
    else:
        log("  Required columns not found. Skipping.")
        test_results['H4'] = {'test': 'skipped', 'reason': 'missing columns'}

    # --- H5: Satisfaction dimensions differ ---
    subsection("H5: Satisfaction scores differ across dimensions")
    log("  H0: Median satisfaction is equal across all measured dimensions")
    log("  H1: At least one dimension has significantly different satisfaction")
    sat_cols = cmap.get('_sat_cols')
    if sat_cols and len(sat_cols) >= 3:
        df_h5 = df[sat_cols].copy()
        for col in sat_cols:
            df_h5[col] = to_numeric_ordinal(df_h5[col])
        df_h5 = df_h5.dropna()

        log(f"\n  Complete cases for Friedman test: {len(df_h5)}")
        for col in sat_cols:
            s = df_h5[col]
            short_name = col[:50]
            log(f"  {short_name:50s}  mean={s.mean():.2f}  std={s.std():.2f}")

        if len(df_h5) >= 5:
            arrays = [df_h5[c].values for c in sat_cols]
            chi2_f, p_f = friedmanchisquare(*arrays)
            k = len(sat_cols)
            W = chi2_f / (len(df_h5) * (k - 1))
            log(f"\n  Friedman chi2({k-1}) = {chi2_f:.3f}, p = {p_f:.4f}")
            log(f"  Kendall's W = {W:.3f} (effect size: {'small' if W < 0.3 else 'medium' if W < 0.5 else 'large'})")
            test_results['H5'] = {'test': 'friedman', 'p': p_f, 'chi2': chi2_f, 'kendalls_W': W}

            if p_f < alpha:
                log(f"\n  Post-hoc pairwise Wilcoxon signed-rank tests (Bonferroni corrected):")
                n_comparisons = k * (k - 1) // 2
                pairs = []
                for i in range(k):
                    for j in range(i + 1, k):
                        w_stat, p_w = stats.wilcoxon(df_h5.iloc[:, i], df_h5.iloc[:, j])
                        p_adj = min(p_w * n_comparisons, 1.0)
                        sig = '*' if p_adj < alpha else ''
                        name_i = sat_cols[i][:25]
                        name_j = sat_cols[j][:25]
                        log(f"    {name_i} vs {name_j}: p_adj={p_adj:.4f} {sig}")
                        pairs.append((sat_cols[i], sat_cols[j], p_adj))
        else:
            log("  Too few complete cases for Friedman test (need >= 5)")
            test_results['H5'] = {'test': 'skipped', 'reason': 'insufficient complete cases'}
    elif sat_cols and len(sat_cols) == 2:
        log("  Only 2 satisfaction columns found, using Wilcoxon signed-rank instead of Friedman")
        df_h5 = df[sat_cols].copy()
        for col in sat_cols:
            df_h5[col] = to_numeric_ordinal(df_h5[col])
        df_h5 = df_h5.dropna()
        if len(df_h5) >= 5:
            w_stat, p_w = stats.wilcoxon(df_h5.iloc[:, 0], df_h5.iloc[:, 1])
            log(f"  Wilcoxon W = {w_stat:.1f}, p = {p_w:.4f}")
            test_results['H5'] = {'test': 'wilcoxon', 'p': p_w}
        else:
            test_results['H5'] = {'test': 'skipped'}
    else:
        log("  Satisfaction columns not found or fewer than 2 detected. Skipping.")
        log("  Check the column mapping above and adjust KEYWORD_MAP if needed.")
        test_results['H5'] = {'test': 'skipped', 'reason': 'missing columns'}

    # --- H6: Weekly budget x trial ---
    subsection("H6: Relationship between weekly food budget and trial")
    log("  H0: Weekly food budget is independent of Mister Maki trial")
    log("  H1: Budget is associated with trial behavior")
    c_budget = cmap.get('weekly_budget')
    if c_budget and c_trial:
        df_h6 = df[[c_budget, c_trial]].dropna()
        df_h6['_trial_bin'] = to_binary(df_h6[c_trial])
        df_h6['_budget_num'] = to_numeric_ordinal(df_h6[c_budget])

        if df_h6['_budget_num'].isna().all():
            def parse_budget_range(val):
                s = str(val)
                nums = re.findall(r'\d+', s)
                if len(nums) >= 2:
                    return (float(nums[0]) + float(nums[1])) / 2
                elif len(nums) == 1:
                    return float(nums[0])
                return np.nan
            df_h6['_budget_num'] = df_h6[c_budget].apply(parse_budget_range)

        df_h6 = df_h6.dropna(subset=['_budget_num'])
        tried = df_h6[df_h6['_trial_bin'] == 1]['_budget_num']
        not_tried = df_h6[df_h6['_trial_bin'] == 0]['_budget_num']

        log(f"\n  Tried (n={len(tried)}):     mean budget = {tried.mean():.1f}, median = {tried.median():.1f}")
        log(f"  Not tried (n={len(not_tried)}): mean budget = {not_tried.mean():.1f}, median = {not_tried.median():.1f}")

        if len(tried) >= 3 and len(not_tried) >= 3:
            u_stat, p_mwu = mannwhitneyu(tried, not_tried, alternative='two-sided')
            r_rb = rank_biserial(u_stat, len(tried), len(not_tried))
            d = cohens_d(tried.values, not_tried.values)
            log(f"\n  Mann-Whitney U = {u_stat:.1f}, p = {p_mwu:.4f}")
            log(f"  Rank-biserial r = {r_rb:.3f}")
            log(f"  Cohen's d = {d:.3f} ({interpret_effect_d(d)})")
            test_results['H6'] = {'test': 'mann_whitney', 'p': p_mwu, 'U': u_stat, 'cohens_d': d}
        else:
            log("  Insufficient data for Mann-Whitney U")
            test_results['H6'] = {'test': 'skipped', 'reason': 'n too small'}

        ct6 = pd.crosstab(df_h6[c_budget], df_h6['_trial_bin'])
        log(f"\n  Cross-tab (budget category x trial):")
        log(f"  {ct6.to_string()}")
    else:
        log("  Required columns not found. Skipping.")
        test_results['H6'] = {'test': 'skipped', 'reason': 'missing columns'}

    # ============================================================
    section("5. CROSS-TABULATIONS")
    # ============================================================

    def cross_tab_report(row_col, col_col, row_label, col_label):
        if row_col is None or col_col is None:
            log(f"  {row_label} x {col_label}: columns not found, skipping")
            return
        ct = pd.crosstab(df[row_col], df[col_col], margins=True, margins_name='Total')
        ct_pct = pd.crosstab(df[row_col], df[col_col], normalize='index') * 100
        log(f"\n  {row_label} x {col_label} (counts):")
        log(f"  {ct.to_string()}")
        log(f"\n  {row_label} x {col_label} (row %):")
        log(f"  {ct_pct.round(1).to_string()}")

    subsection("Trial Rate by Segment")
    cross_tab_report(cmap.get('student_status'), cmap.get('trial'), 'Student Status', 'Trial')

    subsection("Trial Rate by Weekly Budget")
    cross_tab_report(cmap.get('weekly_budget'), cmap.get('trial'), 'Weekly Budget', 'Trial')

    subsection("Trial Rate by Dining Frequency")
    cross_tab_report(cmap.get('dining_freq'), cmap.get('trial'), 'Dining Frequency', 'Trial')

    subsection("Discovery Channel by Student Status")
    cross_tab_report(cmap.get('student_status'), cmap.get('discovery'), 'Student Status', 'Discovery')

    subsection("Awareness by Student Status")
    cross_tab_report(cmap.get('student_status'), cmap.get('awareness'), 'Student Status', 'Awareness')

    # ============================================================
    section("6. CORRELATION MATRIX")
    # ============================================================
    corr_cols_map = {
        'Temaki Familiarity': cmap.get('temaki_familiar'),
        'Value Perception': cmap.get('value_perception'),
        'Reorder Intent': cmap.get('reorder_intent'),
        'Overall Satisfaction': cmap.get('overall_sat'),
    }

    corr_data = {}
    for label, col in corr_cols_map.items():
        if col is not None:
            numeric = to_numeric_ordinal(df[col])
            if numeric.notna().sum() > 0:
                corr_data[label] = numeric

    if len(corr_data) >= 2:
        corr_df = pd.DataFrame(corr_data).dropna()
        log(f"  Variables: {list(corr_data.keys())}")
        log(f"  Complete cases: {len(corr_df)}")

        log(f"\n  Spearman correlation matrix:")
        corr_matrix = corr_df.corr(method='spearman')
        log(f"  {corr_matrix.round(3).to_string()}")

        log(f"\n  P-values:")
        labels = list(corr_data.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                rho, p = spearmanr(corr_df[labels[i]], corr_df[labels[j]])
                sig = '*' if p < 0.05 else ''
                log(f"    {labels[i]} x {labels[j]}: rho={rho:.3f}, p={p:.4f} {sig}")

        fig, ax = plt.subplots(figsize=(6, 5))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        data_masked = corr_matrix.copy()
        data_masked[mask] = np.nan
        im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        short_labels = [l[:15] for l in labels]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
        plt.colorbar(im, ax=ax, label='Spearman rho', shrink=0.8)
        ax.set_title('Correlation Matrix (Spearman)', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig(OUTDIR / 'correlation_matrix.png', bbox_inches='tight')
        plt.close()
        log(f"\n  Saved: {OUTDIR / 'correlation_matrix.png'}")
    else:
        log("  Fewer than 2 numeric variables found for correlation matrix")

    # ============================================================
    section("7. COMPETITIVE POSITIONING")
    # ============================================================
    comp_cols = cmap.get('_comp_cols')
    if comp_cols:
        log(f"  Competitor columns: {comp_cols}")
        comp_data = {}
        for col in comp_cols:
            s = to_numeric_ordinal(df[col]).dropna()
            if len(s) > 0:
                comp_data[col] = {'mean': s.mean(), 'std': s.std(), 'n': len(s), 'median': s.median()}
                log(f"  {col[:60]:60s}  mean={s.mean():.2f}  std={s.std():.2f}  n={len(s)}")

        mm_sat = cmap.get('overall_sat')
        if mm_sat:
            mm_s = to_numeric_ordinal(df[mm_sat]).dropna()
            if len(mm_s) > 0:
                log(f"\n  Mister Maki overall: mean={mm_s.mean():.2f}, std={mm_s.std():.2f}, n={len(mm_s)}")

        if len(comp_data) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = [c[:30] for c in comp_data.keys()]
            means = [comp_data[c]['mean'] for c in comp_data.keys()]
            stds = [comp_data[c]['std'] for c in comp_data.keys()]
            colors = PALETTE[:len(names)]
            bars = ax.barh(range(len(names)), means, xerr=stds, color=colors, edgecolor='none',
                           capsize=3, height=0.6, alpha=0.85)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Mean Rating')
            ax.set_title('Competitive Positioning', fontsize=12, fontweight='bold')
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(m + s + 0.05, i, f'{m:.2f}', va='center', fontsize=8)
            plt.tight_layout()
            plt.savefig(OUTDIR / 'competitive_positioning.png', bbox_inches='tight')
            plt.close()
            log(f"\n  Saved: {OUTDIR / 'competitive_positioning.png'}")
    else:
        log("  No competitor columns detected. Skipping.")

    # ============================================================
    section("8. KEY PROPORTIONS WITH CONFIDENCE INTERVALS")
    # ============================================================
    proportions = {}

    def report_proportion(key, label, positive_check=None):
        col = cmap.get(key)
        if col is None:
            log(f"  {label}: column not found")
            return
        series = df[col].dropna()
        if positive_check:
            pos = series.apply(positive_check).sum()
        else:
            pos = series.apply(lambda v: any(k in str(v).lower() for k in ['yes', 'true', '1'])).sum()
        total = len(series)
        if total == 0:
            return
        p_hat = pos / total
        ci = wilson_ci(p_hat, total)
        log(f"  {label}: {pos}/{total} = {p_hat:.1%}  95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")
        proportions[label] = {'p': p_hat, 'n': total, 'ci_low': ci[0], 'ci_high': ci[1]}

    report_proportion('awareness', 'Brand Awareness (overall)')
    report_proportion('trial', 'Trial Rate (overall)')
    report_proportion('reorder_intent', 'Reorder Intent (among all)')

    if proportions:
        fig, ax = plt.subplots(figsize=(7, 3 + 0.4 * len(proportions)))
        labels = list(proportions.keys())
        ps = [proportions[l]['p'] * 100 for l in labels]
        ci_low = [proportions[l]['ci_low'] * 100 for l in labels]
        ci_high = [proportions[l]['ci_high'] * 100 for l in labels]
        errors = [[p - cl for p, cl in zip(ps, ci_low)],
                  [ch - p for p, ch in zip(ps, ci_high)]]
        y_pos = range(len(labels))
        ax.barh(y_pos, ps, xerr=errors, color=[C1, C2, C3][:len(labels)],
                edgecolor='none', capsize=4, height=0.5, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Percentage (%)')
        ax.set_xlim(0, 105)
        for i, (p, cl, ch) in enumerate(zip(ps, ci_low, ci_high)):
            ax.text(min(p + 3, 95), i, f'{p:.0f}% [{cl:.0f}%, {ch:.0f}%]', va='center', fontsize=8)
        ax.set_title('Key Proportions with 95% Wilson CIs', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTDIR / 'key_proportions.png', bbox_inches='tight')
        plt.close()
        log(f"\n  Saved: {OUTDIR / 'key_proportions.png'}")

    # ============================================================
    section("9. HYPOTHESIS TEST SUMMARY")
    # ============================================================
    log(f"\n  {'Hypothesis':<8} {'Test':<20} {'p-value':<12} {'Effect Size':<20} {'Decision (a=.05)'}")
    log(f"  {'-'*8:<8} {'-'*20:<20} {'-'*12:<12} {'-'*20:<20} {'-'*16}")
    for h_name in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
        r = test_results.get(h_name, {})
        test = r.get('test', 'N/A')
        p = r.get('p', None)
        p_str = f"{p:.4f}" if p is not None else 'N/A'

        if 'cramers_v' in r:
            es = f"V={r['cramers_v']:.3f}"
        elif 'cohens_d' in r:
            es = f"d={r['cohens_d']:.3f}"
        elif 'r' in r:
            es = f"r={r['r']:.3f}"
        elif 'kendalls_W' in r:
            es = f"W={r['kendalls_W']:.3f}"
        elif 'rank_biserial' in r:
            es = f"r_rb={r['rank_biserial']:.3f}"
        else:
            es = 'N/A'

        if p is not None:
            decision = 'Reject H0' if p < alpha else 'Fail to reject'
        else:
            decision = 'Not tested'

        log(f"  {h_name:<8} {test:<20} {p_str:<12} {es:<20} {decision}")

    fig, ax = plt.subplots(figsize=(9, 4))
    h_labels = []
    p_values = []
    colors_p = []
    for h_name in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
        r = test_results.get(h_name, {})
        p = r.get('p', None)
        if p is not None:
            h_labels.append(h_name)
            p_values.append(p)
            colors_p.append(C1 if p < 0.05 else C2)

    if p_values:
        y_pos = range(len(h_labels))
        ax.barh(y_pos, [-np.log10(p) for p in p_values], color=colors_p, edgecolor='none', height=0.5, alpha=0.85)
        ax.axvline(-np.log10(0.05), color='grey', linestyle='--', linewidth=1, label='p = 0.05')
        ax.axvline(-np.log10(0.01), color='grey', linestyle=':', linewidth=1, label='p = 0.01')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(h_labels)
        ax.set_xlabel('-log10(p-value)')
        ax.set_title('Hypothesis Test Results', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        for i, p in enumerate(p_values):
            ax.text(-np.log10(p) + 0.05, i, f'p={p:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTDIR / 'hypothesis_tests.png', bbox_inches='tight')
        plt.close()
        log(f"\n  Saved: {OUTDIR / 'hypothesis_tests.png'}")

    # ============================================================
    section("10. STATISTICAL POWER & LIMITATIONS ANALYSIS")
    # ============================================================

    subsection("Power Analysis")
    log(f"  Actual sample size: n = {n}")
    log(f"  Target sample size: n = 250")
    log(f"  Achievement rate: {n/250:.1%}")
    log(f"")
    log(f"  Minimum detectable effect sizes at 80% power, alpha=0.05:")
    log(f"  (approximations for common tests with n={n})")
    log(f"")

    z_alpha = norm.ppf(0.975)
    z_beta = norm.ppf(0.80)

    d_min = (z_alpha + z_beta) * np.sqrt(4 / n)
    log(f"  Two-sample t-test (equal groups): Cohen's d >= {d_min:.2f} ({interpret_effect_d(d_min)})")

    n_half = n // 2
    w_min = (z_alpha + z_beta) / np.sqrt(n)
    log(f"  Chi-square (1 df): w >= {w_min:.2f} (Cramer's V, {'small' if w_min < 0.3 else 'medium' if w_min < 0.5 else 'large'})")

    r_min = (z_alpha + z_beta) / np.sqrt(n - 3 + (z_alpha + z_beta)**2)
    log(f"  Correlation: |r| >= {r_min:.2f}")

    log(f"")
    log(f"  Practical implication: with n={n}, we can ONLY reliably detect")
    log(f"  large effects. Small-to-medium effects will likely be missed")
    log(f"  (Type II error). Non-significant results should NOT be interpreted")
    log(f"  as evidence of no effect.")

    effect_sizes_range = np.linspace(0.1, 1.5, 50)
    power_values = []
    for d in effect_sizes_range:
        ncp = d * np.sqrt(n / 4)
        power = 1 - norm.cdf(z_alpha - ncp)
        power_values.append(power)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(effect_sizes_range, power_values, color=C1, linewidth=2)
    ax.axhline(0.8, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(d_min, color=C2, linestyle='--', linewidth=1, alpha=0.7)
    ax.text(d_min + 0.02, 0.5, f'd = {d_min:.2f}', color=C2, fontsize=9)
    ax.text(1.3, 0.82, '80% power', color='grey', fontsize=8)
    ax.fill_between(effect_sizes_range, power_values, alpha=0.15, color=C1)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_ylabel('Statistical Power')
    ax.set_title(f'Power Curve (n = {n}, two-sided, alpha = 0.05)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.5)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'power_curve.png', bbox_inches='tight')
    plt.close()
    log(f"  Saved: {OUTDIR / 'power_curve.png'}")

    subsection("Limitations & Threats to Validity")
    log(f"""
  1. SAMPLING BIAS (CRITICAL)
     Sampling method: convenience + snowball sampling via McMaster networks.
     This creates a near-certain over-representation of McMaster students.
     If the student/non-student split is > 70/30, any between-group
     comparisons (H1, H2, H3) have severely unequal group sizes,
     reducing power and potentially violating test assumptions.
     The sample is NOT representative of the broader Hamilton population.

  2. SAMPLE SIZE (CRITICAL)
     n={n} vs. target n=250 means we achieved {100*n/250:.0f}% of planned enrollment.
     Statistical consequences:
     - Minimum detectable Cohen's d ~ {d_min:.2f} (need LARGE effects to reach significance)
     - Chi-square tests with >2 categories will have expected cell counts < 5
     - Confidence intervals for proportions span ~30 percentage points
     - Multiple testing across 6 hypotheses at alpha=0.05 gives family-wise
       error rate of 1-(1-0.05)^6 = {1-(1-0.05)**6:.1%} without correction
     - Bonferroni-corrected alpha = {0.05/6:.4f} makes detection even harder

  3. SELF-SELECTION BIAS
     People who complete restaurant surveys likely have stronger opinions
     (positive or negative) about dining. This truncates the middle of
     the distribution and can inflate satisfaction scores or create
     bimodal patterns that don't reflect the true population.

  4. SOCIAL DESIRABILITY BIAS
     Satisfaction ratings likely skew positive due to:
     - Survey presented as McMaster student project (sympathy effect)
     - Small community (Westdale) where anonymity feels limited
     - General tendency to give 4/5 rather than 2/5 on Likert scales
     Mitigation: look at variance, not just means. High means with
     low variance suggest ceiling effects, not genuine satisfaction.

  5. CROSS-SECTIONAL DESIGN
     Single time point means we cannot establish causality for any
     relationship (e.g., H4: familiarity -> trial could be reversed,
     where trial -> familiarity). All findings are associational only.

  6. MEASUREMENT VALIDITY
     - "Perceived value" may be interpreted differently across demographics
     - "Temaki familiarity" is self-reported (Dunning-Kruger applies)
     - Budget categories may not capture actual spending behavior
     - Discovery channel relies on recall accuracy

  7. NON-RESPONSE BIAS
     With {100*(250-n)/250:.0f}% non-response (vs. target), respondents likely
     differ systematically from non-respondents. Those who responded
     are more likely to be engaged with the local food scene, aware
     of Mister Maki, and connected to McMaster social networks.

  8. MULTIPLE COMPARISONS
     Testing 6 hypotheses inflates Type I error. Bonferroni correction
     (alpha = {0.05/6:.4f}) is conservative but appropriate for this sample.
     Results significant at 0.05 but not at {0.05/6:.4f} should be treated
     as exploratory findings only.
""")

    # ============================================================
    section("11. ADDITIONAL CHARTS")
    # ============================================================

    c_trial = cmap.get('trial')
    c_student = cmap.get('student_status')
    if c_trial and c_student:
        ct_vis = pd.crosstab(df[c_student], df[c_trial])
        fig, ax = plt.subplots(figsize=(7, 4))
        ct_pct = ct_vis.div(ct_vis.sum(axis=1), axis=0) * 100
        ct_pct.plot(kind='bar', ax=ax, color=PALETTE[:ct_pct.shape[1]], edgecolor='none', width=0.7)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Trial Rate by Student Status', fontsize=11, fontweight='bold')
        ax.legend(title='Trial', bbox_to_anchor=(1.05, 1), fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(OUTDIR / 'trial_by_student.png', bbox_inches='tight')
        plt.close()
        log(f"  Saved: {OUTDIR / 'trial_by_student.png'}")

    c_disc = cmap.get('discovery')
    if c_disc:
        vc_disc = df[c_disc].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = PALETTE[:len(vc_disc)]
        bars = ax.barh(range(len(vc_disc)), vc_disc.values, color=colors, edgecolor='none', height=0.6)
        ax.set_yticks(range(len(vc_disc)))
        labels_disc = [str(v)[:40] for v in vc_disc.index]
        ax.set_yticklabels(labels_disc, fontsize=8)
        ax.set_xlabel('Count')
        ax.set_title('Discovery Channels', fontsize=11, fontweight='bold')
        for i, v in enumerate(vc_disc.values):
            ax.text(v + 0.2, i, str(v), va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(OUTDIR / 'discovery_channels.png', bbox_inches='tight')
        plt.close()
        log(f"  Saved: {OUTDIR / 'discovery_channels.png'}")

    sat_cols = cmap.get('_sat_cols')
    if sat_cols and len(sat_cols) >= 2:
        sat_data = {}
        for col in sat_cols:
            s = to_numeric_ordinal(df[col]).dropna()
            if len(s) > 0:
                sat_data[col[:25]] = s
        if len(sat_data) >= 2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bp_data = [sat_data[k].values for k in sat_data.keys()]
            bp = ax.boxplot(bp_data, patch_artist=True, labels=list(sat_data.keys()),
                           widths=0.5, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor=C2, markersize=6))
            for patch, color in zip(bp['boxes'], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel('Rating')
            ax.set_title('Satisfaction Dimensions', fontsize=11, fontweight='bold')
            ax.set_xticklabels(list(sat_data.keys()), rotation=30, ha='right', fontsize=8)
            plt.tight_layout()
            plt.savefig(OUTDIR / 'satisfaction_dimensions.png', bbox_inches='tight')
            plt.close()
            log(f"  Saved: {OUTDIR / 'satisfaction_dimensions.png'}")

    c_budget = cmap.get('weekly_budget')
    if c_budget and c_trial:
        ct_budget = pd.crosstab(df[c_budget], df[c_trial])
        fig, ax = plt.subplots(figsize=(7, 4))
        ct_budget_pct = ct_budget.div(ct_budget.sum(axis=1), axis=0) * 100
        ct_budget_pct.plot(kind='bar', stacked=True, ax=ax, color=PALETTE[:ct_budget_pct.shape[1]],
                          edgecolor='none', width=0.7)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Trial Rate by Weekly Food Budget', fontsize=11, fontweight='bold')
        ax.legend(title='Trial', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTDIR / 'trial_by_budget.png', bbox_inches='tight')
        plt.close()
        log(f"  Saved: {OUTDIR / 'trial_by_budget.png'}")

    # ============================================================
    section("12. RECOMMENDATIONS FOR PRESENTATION")
    # ============================================================
    log(f"""
  WHAT TO EMPHASIZE:
  - Lead with descriptive stats (awareness rate, trial rate with CIs)
  - Frame hypothesis tests as "exploratory" given sample constraints
  - Present effect sizes ALONGSIDE p-values (effect size matters more
    with small n, since p-values are almost entirely a function of n)
  - The limitations section is your credibility section. A prof grading
    a Commerce 3MA3 report wants to see you understand what the data
    CAN'T tell you, because that shows methodological sophistication

  WHAT TO DOWNPLAY:
  - Any p-value between 0.05 and 0.10 should be called "marginally
    significant" or "trending" and NOT used as primary evidence
  - Avoid causal language entirely. "Associated with" not "causes"
  - Don't over-interpret non-significant results as "no effect exists"

  BONFERRONI CORRECTION:
  With 6 tests, adjusted alpha = {0.05/6:.4f}
  Only results with p < {0.05/6:.4f} survive correction.
  Present both corrected and uncorrected for transparency.

  EFFECT SIZE BENCHMARKS (Cohen, 1988):
  - Small:   d=0.2, r=0.1, V=0.1
  - Medium:  d=0.5, r=0.3, V=0.3
  - Large:   d=0.8, r=0.5, V=0.5
  With n={n}, we need d >= {d_min:.2f} to detect at 80% power.
""")

    section("ANALYSIS COMPLETE")
    log(f"  Output directory: {OUTDIR}")
    log(f"  Results file: {RESULTS_FILE}")

    with open(RESULTS_FILE, 'w') as f:
        f.write('\n'.join(results))

    log(f"\n  All results saved to {RESULTS_FILE}")

    chart_files = list(OUTDIR.glob('*.png'))
    log(f"  Charts generated: {len(chart_files)}")
    for cf in sorted(chart_files):
        log(f"    - {cf.name}")


if __name__ == '__main__':
    run_analysis()
