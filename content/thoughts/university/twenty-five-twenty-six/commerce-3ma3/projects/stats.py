#!/usr/bin/env python3

import warnings

warnings.filterwarnings('ignore')

import pathlib
import re, os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

DATA = pathlib.Path(__file__).parent / 'MisterMaki_Survey_Response.xlsx'
OUT = pathlib.Path(os.getenv("OUT_PATH", '/tmp/mister_maki_charts'))
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA)
N = len(df)


# ── font setup ──────────────────────────────────────────────────────────
def _pick_font(candidates, fallback):
  available = {f.name for f in fm.fontManager.ttflist}
  for c in candidates:
    if c in available:
      return c
  return fallback


FONT_BODY = _pick_font(['Libre Franklin', 'DejaVu Sans'], 'sans-serif')
FONT_MONO = _pick_font(['DM Mono', 'DejaVu Sans Mono'], 'monospace')

plt.rcParams.update({
  'font.family': FONT_BODY,
  'axes.facecolor': '#FFFEE0',
  'figure.facecolor': '#FFFEE0',
  'axes.edgecolor': 'none',
  'axes.grid': False,
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.spines.bottom': False,
  'axes.spines.left': False,
  'xtick.major.size': 0,
  'ytick.major.size': 0,
  'text.color': '#000000',
})

CREAM = '#FFFEE0'
PALE_YELLOW = '#FEF9C3'
LIGHT_BLUE = '#DBEAFE'
LAVENDER = '#E9D5FF'
GREEN = '#BEE3A0'
PINK = '#FECDD3'
GRAY = '#E5E7EB'
PALETTE = [PALE_YELLOW, LIGHT_BLUE, LAVENDER, GREEN, PINK, GRAY]

# ── column extraction helpers ───────────────────────────────────────────
segment = df.iloc[:, 6]
is_student = segment == 'McMaster student'
is_resident = segment == 'Hamilton resident (not a McMaster student)'

dining_freq = df.iloc[:, 9]
budget = df.iloc[:, 10]
discovery_channels = df.iloc[:, 11]
aided_awareness = df.iloc[:, 13]
first_learned = df.iloc[:, 14]
temaki_fam_raw = df.iloc[:, 15]
ever_temaki = df.iloc[:, 16]
mm_trial = df.iloc[:, 17]

sat_cols = {
  'Food quality & freshness': df.iloc[:, 19],
  'Portion size vs price': df.iloc[:, 20],
  'Hand roll assembly ease': df.iloc[:, 21],
  'Packaging & presentation': df.iloc[:, 22],
  'Speed of service/delivery': df.iloc[:, 23],
}
reorder_raw = df.iloc[:, 24]
value_raw = df.iloc[:, 25]

# ── encoding functions ──────────────────────────────────────────────────
SAT_MAP = {'Very Satisfied': 5, 'Satisfied': 4, 'Neutral': 3, 'Dissatisfied': 2, 'Very Dissatisfied': 1}

BUDGET_MAP = {'Less than $20': 1, '$20 - $40': 2, '$41 - $60': 3, '$61 - $80': 4, 'More than $80': 5}


def leading_digit(s):
  if pd.isna(s):
    return np.nan
  m = re.match(r'(\d)', str(s))
  return int(m.group(1)) if m else np.nan


temaki_fam = temaki_fam_raw.apply(leading_digit)
reorder_intent = reorder_raw.apply(leading_digit)
value_perc = value_raw.apply(leading_digit)
trial_binary = (mm_trial == 'Yes').astype(int)

sat_encoded = pd.DataFrame({k: v.map(SAT_MAP) for k, v in sat_cols.items()})
budget_ord = budget.map(BUDGET_MAP)

mm_aware = aided_awareness.fillna('').str.contains('Mister Maki', case=False).astype(int)

is_wom = (first_learned == 'Friend / classmate / family').astype(int)


# ── helper: Wilson CI ───────────────────────────────────────────────────
def wilson_ci(count, nobs, alpha=0.05):
  lo, hi = proportion_confint(count, nobs, alpha=alpha, method='wilson')
  return lo, hi


# ── helper: Cramer's V ─────────────────────────────────────────────────
def cramers_v(table):
  chi2 = stats.chi2_contingency(table, correction=False)[0]
  t = np.array(table)
  r, k = t.shape
  return np.sqrt(chi2 / (t.sum() * (min(r, k) - 1)))


# ── helper: rank-biserial r ────────────────────────────────────────────
def rank_biserial(u, n1, n2):
  return 1 - (2 * u) / (n1 * n2)


# ── results accumulator ────────────────────────────────────────────────
results = []
report_lines = []


def log(text):
  report_lines.append(text)


log('=' * 80)
log('MISTER MAKI SURVEY — STATISTICAL ANALYSIS v2')
log(f'N = {N}  |  Students = {is_student.sum()}  |  Residents = {is_resident.sum()}')
log(f'MM Customers (trial=Yes) = {(mm_trial == "Yes").sum()}  |  Non-customers = {(mm_trial == "No").sum()}')
log('=' * 80)
log('')

# ── H1: Aided awareness by segment ─────────────────────────────────────
log('─' * 80)
log('H1: Mister Maki aided awareness differs between students and residents')

a_stu = mm_aware[is_student].sum()
n_stu = is_student.sum()
a_res = mm_aware[is_resident].sum()
n_res = is_resident.sum()
table_h1 = np.array([[a_stu, n_stu - a_stu], [a_res, n_res - a_res]])
_, p_h1 = stats.fisher_exact(table_h1)
v_h1 = cramers_v(table_h1)
ci_stu = wilson_ci(a_stu, n_stu)
ci_res = wilson_ci(a_res, n_res)

log('  H0: Proportion aware of MM is equal across segments')
log('  H1: Proportion aware of MM differs by segment')
log("  Test: Fisher's exact (2×2, small cells)")
log(f'  Students aware: {a_stu}/{n_stu} = {a_stu / n_stu:.1%}  CI95 [{ci_stu[0]:.1%}, {ci_stu[1]:.1%}]')
log(f'  Residents aware: {a_res}/{n_res} = {a_res / n_res:.1%}  CI95 [{ci_res[0]:.1%}, {ci_res[1]:.1%}]')
log(f"  p = {p_h1:.4f}  |  Cramér's V = {v_h1:.3f}")
log(f'  Decision: {"Reject H0" if p_h1 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H1',
  'Aided awareness × segment',
  'Fisher exact',
  f'p={p_h1:.4f}',
  f'V={v_h1:.3f}',
  'Reject' if p_h1 < 0.05 else 'Fail to reject',
))
log('')

# ── H2: WOM discovery by segment ───────────────────────────────────────
log('─' * 80)
log('H2: Word-of-mouth discovery differs by segment')

valid_h2 = first_learned.notna() & ~first_learned.str.lower().str.contains(
  "don't know|never have|don't know", na=False
)
wom_stu = is_wom[is_student & valid_h2].sum()
n_stu_h2 = (is_student & valid_h2).sum()
wom_res = is_wom[is_resident & valid_h2].sum()
n_res_h2 = (is_resident & valid_h2).sum()
table_h2 = np.array([[wom_stu, n_stu_h2 - wom_stu], [wom_res, n_res_h2 - wom_res]])
_, p_h2 = stats.fisher_exact(table_h2)
v_h2 = cramers_v(table_h2)

log('  H0: WOM as first source is equal across segments')
log('  H1: WOM as first source differs by segment')
log("  Test: Fisher's exact (2×2)")
log(f'  Students WOM: {wom_stu}/{n_stu_h2} = {wom_stu / n_stu_h2:.1%}')
log(f'  Residents WOM: {wom_res}/{n_res_h2} = {wom_res / n_res_h2:.1%}')
log(f"  p = {p_h2:.4f}  |  Cramér's V = {v_h2:.3f}")
log(f'  Decision: {"Reject H0" if p_h2 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H2',
  'WOM discovery × segment',
  'Fisher exact',
  f'p={p_h2:.4f}',
  f'V={v_h2:.3f}',
  'Reject' if p_h2 < 0.05 else 'Fail to reject',
))
log('')

# ── H3: Value perception by segment ────────────────────────────────────
log('─' * 80)
log('H3: Value perception differs by segment')

vp_stu = value_perc[is_student].dropna()
vp_res = value_perc[is_resident].dropna()
if len(vp_stu) > 0 and len(vp_res) > 0:
  u_h3, p_h3 = stats.mannwhitneyu(vp_stu, vp_res, alternative='two-sided')
  rb_h3 = rank_biserial(u_h3, len(vp_stu), len(vp_res))
  log('  H0: Value perception distribution is equal across segments')
  log('  H1: Value perception distribution differs by segment')
  log('  Test: Mann-Whitney U (ordinal data)')
  log(f'  Students: median={vp_stu.median():.0f}, mean={vp_stu.mean():.2f}, n={len(vp_stu)}')
  log(f'  Residents: median={vp_res.median():.0f}, mean={vp_res.mean():.2f}, n={len(vp_res)}')
  log(f'  U = {u_h3:.1f}  |  p = {p_h3:.4f}  |  rank-biserial r = {rb_h3:.3f}')
  log(f'  Decision: {"Reject H0" if p_h3 < 0.05 else "Fail to reject H0"} at α=0.05')
  results.append((
    'H3',
    'Value perception × segment',
    'Mann-Whitney U',
    f'U={u_h3:.1f}, p={p_h3:.4f}',
    f'r={rb_h3:.3f}',
    'Reject' if p_h3 < 0.05 else 'Fail to reject',
  ))
else:
  log('  Insufficient data for comparison')
  results.append(('H3', 'Value perception × segment', 'Mann-Whitney U', 'N/A', 'N/A', 'N/A'))
log('')

# ── H4: Temaki familiarity × MM trial ──────────────────────────────────
log('─' * 80)
log('H4: Temaki familiarity associated with MM trial')

valid_h4 = temaki_fam.notna() & mm_trial.notna()
fam_v = temaki_fam[valid_h4]
trial_v = trial_binary[valid_h4]

rpb, p_rpb = stats.pointbiserialr(trial_v, fam_v)
rho, p_rho = stats.spearmanr(fam_v, trial_v)

log('  H0: No association between temaki familiarity and MM trial')
log('  H1: Higher temaki familiarity is associated with MM trial')
log('  Test: Point-biserial r (continuous × binary)')
log(f'  r_pb = {rpb:.3f}  |  p = {p_rpb:.4f}')
log(f'  Robustness: Spearman ρ = {rho:.3f}  |  p = {p_rho:.4f}')
log(f'  Decision: {"Reject H0" if p_rpb < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H4',
  'Temaki fam. × trial',
  'Point-biserial',
  f'r={rpb:.3f}, p={p_rpb:.4f}',
  f'ρ={rho:.3f}',
  'Reject' if p_rpb < 0.05 else 'Fail to reject',
))
log('')

# ── H5: Satisfaction across 5 dimensions (Friedman) ────────────────────
log('─' * 80)
log('H5: Satisfaction differs across 5 dimensions (customers only)')

sat_df = sat_encoded.dropna()
n_cust = len(sat_df)
log(f'  n = {n_cust} customers with complete satisfaction data')

means = sat_df.mean()
for dim, val in means.items():
  log(f'    {dim}: mean = {val:.2f}')

if n_cust >= 3:
  fr_stat, p_fr = stats.friedmanchisquare(*[sat_df[c] for c in sat_df.columns])
  w_kendall = fr_stat / (n_cust * (len(sat_df.columns) - 1))
  log('  Test: Friedman χ² (repeated-measures ordinal)')
  log(f"  χ² = {fr_stat:.3f}  |  p = {p_fr:.4f}  |  Kendall's W = {w_kendall:.3f}")
  log(f'  Decision: {"Reject H0" if p_fr < 0.05 else "Fail to reject H0"} at α=0.05')

  if p_fr < 0.05:
    log('  Post-hoc Wilcoxon signed-rank with Bonferroni correction:')
    cols = list(sat_df.columns)
    n_comparisons = len(cols) * (len(cols) - 1) // 2
    for i in range(len(cols)):
      for j in range(i + 1, len(cols)):
        w_stat, p_w = stats.wilcoxon(sat_df[cols[i]], sat_df[cols[j]], zero_method='wilcox', alternative='two-sided')
        p_adj = min(p_w * n_comparisons, 1.0)
        sig = '*' if p_adj < 0.05 else ''
        log(f'    {cols[i]} vs {cols[j]}: W={w_stat:.0f}, p_adj={p_adj:.4f} {sig}')

  results.append((
    'H5',
    'Satisfaction across dims',
    'Friedman',
    f'χ²={fr_stat:.3f}, p={p_fr:.4f}',
    f'W={w_kendall:.3f}',
    'Reject' if p_fr < 0.05 else 'Fail to reject',
  ))
else:
  log('  Too few customers for Friedman test')
  results.append(('H5', 'Satisfaction across dims', 'Friedman', 'N/A', 'N/A', 'N/A'))
log('')

# ── H6: Budget × trial ─────────────────────────────────────────────────
log('─' * 80)
log('H6: Weekly budget associated with MM trial')

valid_h6 = budget_ord.notna() & mm_trial.notna()
bud_yes = budget_ord[valid_h6 & (mm_trial == 'Yes')]
bud_no = budget_ord[valid_h6 & (mm_trial == 'No')]

u_h6, p_h6 = stats.mannwhitneyu(bud_yes, bud_no, alternative='two-sided')
rb_h6 = rank_biserial(u_h6, len(bud_yes), len(bud_no))

log('  H0: Budget distribution is equal for trial=Yes vs trial=No')
log('  H1: Budget distribution differs by trial status')
log('  Test: Mann-Whitney U')
log(f'  Trial=Yes: median={bud_yes.median():.0f}, mean={bud_yes.mean():.2f}, n={len(bud_yes)}')
log(f'  Trial=No:  median={bud_no.median():.0f}, mean={bud_no.mean():.2f}, n={len(bud_no)}')
log(f'  U = {u_h6:.1f}  |  p = {p_h6:.4f}  |  rank-biserial r = {rb_h6:.3f}')
log(f'  Decision: {"Reject H0" if p_h6 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H6',
  'Budget × trial',
  'Mann-Whitney U',
  f'U={u_h6:.1f}, p={p_h6:.4f}',
  f'r={rb_h6:.3f}',
  'Reject' if p_h6 < 0.05 else 'Fail to reject',
))
log('')

# ── H7: Dining frequency × trial ───────────────────────────────────────
log('─' * 80)
log('H7: Dining frequency associated with MM trial')

freq_regular = dining_freq.isin(['Regularly (1-2 times a week)', 'Frequently (3+ times a week)'])
reg_trial = (freq_regular & (mm_trial == 'Yes')).sum()
reg_no = (freq_regular & (mm_trial == 'No')).sum()
occ_trial = (~freq_regular & (mm_trial == 'Yes')).sum()
occ_no = (~freq_regular & (mm_trial == 'No')).sum()
table_h7 = np.array([[reg_trial, reg_no], [occ_trial, occ_no]])
_, p_h7 = stats.fisher_exact(table_h7)
v_h7 = cramers_v(table_h7)
total_reg = reg_trial + reg_no
total_occ = occ_trial + occ_no
ci_reg = wilson_ci(reg_trial, total_reg) if total_reg > 0 else (0, 0)
ci_occ = wilson_ci(occ_trial, total_occ) if total_occ > 0 else (0, 0)

log('  H0: Trial rate is independent of dining frequency')
log('  H1: Regular+ diners have different trial rate than occasional-')
log("  Test: Fisher's exact (2×2)")
log(
  f'  Regular+ diners trial rate: {reg_trial}/{total_reg} = {reg_trial / total_reg:.1%}  CI95 [{ci_reg[0]:.1%}, {ci_reg[1]:.1%}]'
)
log(
  f'  Occasional- diners trial rate: {occ_trial}/{total_occ} = {occ_trial / total_occ:.1%}  CI95 [{ci_occ[0]:.1%}, {ci_occ[1]:.1%}]'
)
log(f"  p = {p_h7:.4f}  |  Cramér's V = {v_h7:.3f}")
log(f'  Decision: {"Reject H0" if p_h7 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H7',
  'Dining freq × trial',
  'Fisher exact',
  f'p={p_h7:.4f}',
  f'V={v_h7:.3f}',
  'Reject' if p_h7 < 0.05 else 'Fail to reject',
))
log('')

# ── H8: Reorder intent × mean satisfaction ──────────────────────────────
log('─' * 80)
log('H8: Reorder intent correlates with mean satisfaction (customers only)')

cust_mask = mm_trial == 'Yes'
mean_sat = sat_encoded[cust_mask].mean(axis=1)
reorder_cust = reorder_intent[cust_mask]
valid_h8 = mean_sat.notna() & reorder_cust.notna()

rho_h8, p_h8 = stats.spearmanr(mean_sat[valid_h8], reorder_cust[valid_h8])
log('  H0: No correlation between mean satisfaction and reorder intent')
log('  H1: Positive correlation exists')
log('  Test: Spearman rank correlation (ordinal)')
log(f'  n = {valid_h8.sum()}  |  ρ = {rho_h8:.3f}  |  p = {p_h8:.4f}')
log(f'  Decision: {"Reject H0" if p_h8 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H8',
  'Reorder × satisfaction',
  'Spearman',
  f'ρ={rho_h8:.3f}, p={p_h8:.4f}',
  f'ρ={rho_h8:.3f}',
  'Reject' if p_h8 < 0.05 else 'Fail to reject',
))
log('')

# ── H9: Temaki familiarity × trial (Mann-Whitney) ──────────────────────
log('─' * 80)
log('H9: Customers vs non-customers differ in temaki familiarity')

fam_yes = temaki_fam[mm_trial == 'Yes'].dropna()
fam_no = temaki_fam[mm_trial == 'No'].dropna()
u_h9, p_h9 = stats.mannwhitneyu(fam_yes, fam_no, alternative='two-sided')
rb_h9 = rank_biserial(u_h9, len(fam_yes), len(fam_no))

log('  H0: Temaki familiarity distribution is equal for trial=Yes vs No')
log('  H1: Temaki familiarity differs by trial status')
log('  Test: Mann-Whitney U')
log(f'  Trial=Yes: median={fam_yes.median():.0f}, mean={fam_yes.mean():.2f}, n={len(fam_yes)}')
log(f'  Trial=No:  median={fam_no.median():.0f}, mean={fam_no.mean():.2f}, n={len(fam_no)}')
log(f'  U = {u_h9:.1f}  |  p = {p_h9:.4f}  |  rank-biserial r = {rb_h9:.3f}')
log(f'  Decision: {"Reject H0" if p_h9 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H9',
  'Temaki fam. (cust vs non)',
  'Mann-Whitney U',
  f'U={u_h9:.1f}, p={p_h9:.4f}',
  f'r={rb_h9:.3f}',
  'Reject' if p_h9 < 0.05 else 'Fail to reject',
))
log('')

# ── H10: Discovery source × trial ──────────────────────────────────────
log('─' * 80)
log('H10: Discovery source (WOM vs walk-by) associated with trial')

valid_sources = ['Friend / classmate / family', 'Walked or drove past']
h10_mask = first_learned.isin(valid_sources) & mm_trial.notna()
h10_df = df[h10_mask]
wom_trial = ((h10_df.iloc[:, 14] == 'Friend / classmate / family') & (h10_df.iloc[:, 17] == 'Yes')).sum()
wom_notrial = ((h10_df.iloc[:, 14] == 'Friend / classmate / family') & (h10_df.iloc[:, 17] == 'No')).sum()
walk_trial = ((h10_df.iloc[:, 14] == 'Walked or drove past') & (h10_df.iloc[:, 17] == 'Yes')).sum()
walk_notrial = ((h10_df.iloc[:, 14] == 'Walked or drove past') & (h10_df.iloc[:, 17] == 'No')).sum()

table_h10 = np.array([[wom_trial, wom_notrial], [walk_trial, walk_notrial]])
_, p_h10 = stats.fisher_exact(table_h10)
v_h10 = cramers_v(table_h10)

log('  H0: Trial rate is independent of discovery channel (WOM vs walk-by)')
log('  H1: Trial rate differs by discovery channel')
log("  Test: Fisher's exact (2×2, drop rare categories)")
n_wom = wom_trial + wom_notrial
n_walk = walk_trial + walk_notrial
log(f'  WOM trial: {wom_trial}/{n_wom} = {wom_trial / n_wom:.1%}' if n_wom > 0 else '  WOM: n=0')
log(f'  Walk-by trial: {walk_trial}/{n_walk} = {walk_trial / n_walk:.1%}' if n_walk > 0 else '  Walk-by: n=0')
log(f"  p = {p_h10:.4f}  |  Cramér's V = {v_h10:.3f}")
log(f'  Decision: {"Reject H0" if p_h10 < 0.05 else "Fail to reject H0"} at α=0.05')
results.append((
  'H10',
  'Discovery × trial',
  'Fisher exact',
  f'p={p_h10:.4f}',
  f'V={v_h10:.3f}',
  'Reject' if p_h10 < 0.05 else 'Fail to reject',
))
log('')

# ── summary table ───────────────────────────────────────────────────────
log('=' * 80)
log('SUMMARY TABLE')
log('=' * 80)
header = f'{"Hyp":<5} {"Description":<28} {"Test":<18} {"Stat / p-value":<26} {"Effect":<12} {"Decision"}'
log(header)
log('-' * len(header))
for r in results:
  log(f'{r[0]:<5} {r[1]:<28} {r[2]:<18} {r[3]:<26} {r[4]:<12} {r[5]}')
log('')

# ── save text ───────────────────────────────────────────────────────────
report_text = '\n'.join(report_lines)
(OUT / 'analysis_results.txt').write_text(report_text)
print(report_text)


# ════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════

DPI = 200
FIG_W, FIG_H = 1600 / DPI, 900 / DPI


def save(fig, name):
  fig.savefig(OUT / name, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
  plt.close(fig)
  print(f'  saved → {OUT / name}')


# ── 1. Hypothesis summary table ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W * 1.3, FIG_H * 1.1))
fig.set_facecolor(CREAM)
ax.set_facecolor(CREAM)
ax.axis('off')

col_labels = ['Hypothesis', 'Description', 'Test', 'Stat / p-value', 'Effect Size', 'Decision']
cell_text = [list(r) for r in results]

table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='left', colLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
  cell.set_edgecolor('#CCCCCC')
  cell.set_linewidth(0.3)
  if row == 0:
    cell.set_facecolor('#E5E7EB')
    cell.set_text_props(fontfamily=FONT_MONO, fontweight='bold', fontsize=7)
  else:
    cell.set_facecolor(CREAM)
    cell.set_text_props(fontfamily=FONT_MONO, fontsize=6.5)
    decision = cell_text[row - 1][5]
    if 'Reject' == decision.strip():
      cell.set_facecolor('#FEF9C3')

fig.suptitle('HYPOTHESIS TESTING SUMMARY', fontfamily=FONT_MONO, fontsize=11, fontweight='bold', y=0.96)
save(fig, 'hypothesis_summary.png')

# ── 2. Trial by segment ────────────────────────────────────────────────
trial_stu = (mm_trial[is_student] == 'Yes').sum()
trial_res = (mm_trial[is_resident] == 'Yes').sum()
pct_stu = trial_stu / is_student.sum() * 100
pct_res = trial_res / is_resident.sum() * 100

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

bars = ax.barh(
  ['Hamilton Residents', 'McMaster Students'],
  [pct_res, pct_stu],
  color=[LAVENDER, LIGHT_BLUE],
  height=0.5,
  edgecolor='none',
)

for bar, pct, n_grp, n_trial in zip(
  bars, [pct_res, pct_stu], [is_resident.sum(), is_student.sum()], [trial_res, trial_stu]
):
  ax.text(
    bar.get_width() + 1.5,
    bar.get_y() + bar.get_height() / 2,
    f'{pct:.1f}%  ({n_trial}/{n_grp})',
    va='center',
    fontsize=14,
    fontweight='bold',
    fontfamily=FONT_BODY,
  )

ax.set_xlim(0, 80)
ax.set_xticks([])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Hamilton Residents', 'McMaster Students'], fontsize=12, fontfamily=FONT_BODY)
ax.invert_yaxis()
fig.suptitle('MISTER MAKI TRIAL RATE BY SEGMENT', fontfamily=FONT_MONO, fontsize=13, fontweight='bold', y=0.95)
ax.text(
  0,
  -0.6,
  f"Fisher's exact p = {p_h7:.3f}",
  fontsize=9,
  fontfamily=FONT_MONO,
  color='#666666',
  transform=ax.get_yaxis_transform(),
)
save(fig, 'trial_by_segment.png')

# ── 3. Satisfaction bars ────────────────────────────────────────────────
sat_means = sat_encoded.dropna().mean().sort_values()

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

colors = [PINK if 'Portion' in dim else LIGHT_BLUE for dim in sat_means.index]
bars = ax.barh(range(len(sat_means)), sat_means.values, color=colors, height=0.55, edgecolor='none')

for i, (bar, val) in enumerate(zip(bars, sat_means.values)):
  ax.text(
    bar.get_width() + 0.05,
    bar.get_y() + bar.get_height() / 2,
    f'{val:.2f}',
    va='center',
    fontsize=12,
    fontweight='bold',
    fontfamily=FONT_BODY,
  )

ax.set_yticks(range(len(sat_means)))
ax.set_yticklabels(sat_means.index, fontsize=10, fontfamily=FONT_BODY)
ax.set_xlim(0, 5.5)
ax.set_xticks([])
fig.suptitle('CUSTOMER SATISFACTION BY DIMENSION', fontfamily=FONT_MONO, fontsize=13, fontweight='bold', y=0.95)
ax.text(
  0.01,
  -0.05,
  f'n = {n_cust} customers  |  Friedman χ² = {fr_stat:.2f}, p = {p_fr:.4f}',
  fontsize=9,
  fontfamily=FONT_MONO,
  color='#666666',
  transform=ax.transAxes,
)
save(fig, 'satisfaction.png')

# ── 4. Familiarity → trial rate ─────────────────────────────────────────
fam_trial_df = pd.DataFrame({'fam': temaki_fam, 'trial': trial_binary}).dropna()
fam_groups = fam_trial_df.groupby('fam')['trial']
fam_rates = fam_groups.mean() * 100
fam_counts = fam_groups.size()

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

x = fam_rates.index.astype(int)
bars = ax.bar(x, fam_rates.values, color=[PALETTE[int(i) % len(PALETTE)] for i in x], width=0.6, edgecolor='none')

for bar, rate, n in zip(bars, fam_rates.values, fam_counts.values):
  ax.text(
    bar.get_x() + bar.get_width() / 2,
    bar.get_height() + 2,
    f'{rate:.0f}%\n(n={n})',
    ha='center',
    va='bottom',
    fontsize=11,
    fontweight='bold',
    fontfamily=FONT_BODY,
  )

ax.set_xticks(x)
ax.set_xticklabels([f'{int(v)}' for v in x], fontsize=11, fontfamily=FONT_BODY)
ax.set_xlabel('Temaki Familiarity (1=None → 5=Regular eater)', fontsize=10, fontfamily=FONT_BODY)
ax.set_ylim(0, 110)
ax.set_yticks([])
fig.suptitle('TRIAL RATE BY TEMAKI FAMILIARITY', fontfamily=FONT_MONO, fontsize=13, fontweight='bold', y=0.95)
ax.text(
  0.01,
  -0.08,
  f'Point-biserial r = {rpb:.3f}, p = {p_rpb:.4f}',
  fontsize=9,
  fontfamily=FONT_MONO,
  color='#666666',
  transform=ax.transAxes,
)
save(fig, 'familiarity_trial.png')

# ── 5. Dining frequency → trial ────────────────────────────────────────
freq_labels_ordered = [
  'Never',
  'Rarely (less than once a month)',
  'Occasionally (1-2 times a month)',
  'Regularly (1-2 times a week)',
  'Frequently (3+ times a week)',
]
freq_trial_df = pd.DataFrame({'freq': dining_freq, 'trial': trial_binary}).dropna()
freq_trial_df['freq'] = pd.Categorical(freq_trial_df['freq'], categories=freq_labels_ordered, ordered=True)
freq_grp = freq_trial_df.groupby('freq', observed=False)['trial']
freq_rates = freq_grp.mean() * 100
freq_ns = freq_grp.size()

short_labels = ['Never', 'Rarely\n(<1x/mo)', 'Occasionally\n(1-2x/mo)', 'Regularly\n(1-2x/wk)', 'Frequently\n(3+/wk)']

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

bars = ax.bar(
  range(len(freq_rates)),
  freq_rates.values,
  color=[PALETTE[i % len(PALETTE)] for i in range(len(freq_rates))],
  width=0.6,
  edgecolor='none',
)

for bar, rate, n in zip(bars, freq_rates.values, freq_ns.values):
  if n > 0:
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 2,
      f'{rate:.0f}%\n(n={n})',
      ha='center',
      va='bottom',
      fontsize=10,
      fontweight='bold',
      fontfamily=FONT_BODY,
    )

ax.set_xticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, fontsize=9, fontfamily=FONT_BODY)
ax.set_ylim(0, 110)
ax.set_yticks([])
fig.suptitle(
  'MISTER MAKI TRIAL RATE BY DINING FREQUENCY', fontfamily=FONT_MONO, fontsize=13, fontweight='bold', y=0.95
)
ax.text(
  0.01,
  -0.1,
  f"Fisher's exact p = {p_h7:.4f}  |  Cramér's V = {v_h7:.3f}",
  fontsize=9,
  fontfamily=FONT_MONO,
  color='#666666',
  transform=ax.transAxes,
)
save(fig, 'dining_trial.png')

# ── 6. Value perception by segment ─────────────────────────────────────
vp_all = pd.DataFrame({'segment': segment, 'value': value_perc}).dropna()
vp_stu_dist = vp_all[vp_all['segment'] == 'McMaster student']['value'].value_counts().sort_index()
vp_res_dist = (
  vp_all[vp_all['segment'] == 'Hamilton resident (not a McMaster student)']['value'].value_counts().sort_index()
)

all_vals = [1, 2, 3, 4, 5]
vp_stu_pct = pd.Series([vp_stu_dist.get(v, 0) for v in all_vals], index=all_vals)
vp_res_pct = pd.Series([vp_res_dist.get(v, 0) for v in all_vals], index=all_vals)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

width = 0.35
x = np.array(all_vals)
b1 = ax.bar(x - width / 2, vp_stu_pct.values, width, label='Students', color=LIGHT_BLUE, edgecolor='none')
b2 = ax.bar(x + width / 2, vp_res_pct.values, width, label='Residents', color=LAVENDER, edgecolor='none')

for bar in list(b1) + list(b2):
  h = bar.get_height()
  if h > 0:
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      h + 0.15,
      f'{int(h)}',
      ha='center',
      fontsize=10,
      fontweight='bold',
      fontfamily=FONT_BODY,
    )

ax.set_xticks(all_vals)
ax.set_xticklabels(
  ['1\nPoor', '2\nSomewhat\npoor', '3\nAverage', '4\nSomewhat\ngood', '5\nExcellent'], fontsize=9, fontfamily=FONT_BODY
)
ax.set_yticks([])
ax.legend(fontsize=10, frameon=False)
fig.suptitle('VALUE PERCEPTION BY SEGMENT', fontfamily=FONT_MONO, fontsize=13, fontweight='bold', y=0.95)
stat_txt = (
  f'Mann-Whitney U = {u_h3:.1f}, p = {p_h3:.4f}' if len(vp_stu) > 0 and len(vp_res) > 0 else 'insufficient data'
)
ax.text(0.01, -0.1, stat_txt, fontsize=9, fontfamily=FONT_MONO, color='#666666', transform=ax.transAxes)
save(fig, 'value_by_segment.png')

print('\n✓ all charts saved to', OUT)
