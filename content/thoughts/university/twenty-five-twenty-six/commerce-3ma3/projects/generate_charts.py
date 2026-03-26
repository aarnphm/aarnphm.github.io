import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIGSIZE = (1600 / 200, 900 / 200)
DPI = 200
BG = '#FFFFFF'
TEXT = '#1A1A2E'
STUDENTS = '#4A90D9'
RESIDENTS = '#E8845C'
AMBER = '#F5A623'
TEAL = '#10B981'

plt.rcParams.update({
    'font.family': ['Inter', 'Helvetica Neue', 'Arial', 'sans-serif'],
    'font.size': 9,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'savefig.facecolor': BG,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

OUT = '/tmp/mister_maki_charts'

# ========================================================================
# 1. DONUT CHART - sample_split.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
sizes = [25, 12]
colors = [STUDENTS, RESIDENTS]
labels = ['McMaster Students', 'Hamilton Residents']
pcts = [67.6, 32.4]

wedges, _ = ax.pie(
    sizes, colors=colors, startangle=90, counterclock=False,
    wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=3)
)

for i, (wedge, label, pct, n) in enumerate(zip(wedges, labels, pcts, sizes)):
    ang = (wedge.theta2 + wedge.theta1) / 2
    x = 0.77 * np.cos(np.radians(ang))
    y = 0.77 * np.sin(np.radians(ang))
    ax.text(x, y, f'{pct:.0f}%', ha='center', va='center',
            fontsize=14, fontweight='bold', color=BG)

ax.text(0, 0.04, 'n = 37', ha='center', va='center',
        fontsize=16, fontweight='bold', color=TEXT)
ax.text(0, -0.1, 'respondents', ha='center', va='center',
        fontsize=8, color='#666666')

ax.set_title('Survey Sample Composition', fontsize=14, fontweight='bold',
             color=TEXT, pad=20)
ax.set_aspect('equal')

fig.text(0.30, 0.08, '\u25CF', fontsize=14, color=STUDENTS, ha='center', va='center')
fig.text(0.335, 0.08, f'McMaster Students (n=25)', fontsize=8, color=TEXT, ha='left', va='center')
fig.text(0.62, 0.08, '\u25CF', fontsize=14, color=RESIDENTS, ha='center', va='center')
fig.text(0.655, 0.08, f'Hamilton Residents (n=12)', fontsize=8, color=TEXT, ha='left', va='center')

fig.savefig(f'{OUT}/sample_split.png')
plt.close()

# ========================================================================
# 2. STAT CARDS - awareness_stats.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

stats = [
    ('54%', 'Unaided Recall', 'Top restaurant named\nwithout prompting', STUDENTS),
    ('81%', 'Aided Awareness', 'Recognized Mister Maki\nfrom a list', STUDENTS),
    ('89%', 'Word-of-Mouth\nDiscovery', 'Among those aware,\nlearned organically', TEAL),
]

card_width = 3.2
gap = 0.6
total_width = 3 * card_width + 2 * gap
start_x = (12 - total_width) / 2

for i, (num, title, desc, color) in enumerate(stats):
    cx = start_x + i * (card_width + gap) + card_width / 2
    cy = 3.0

    rect = FancyBboxPatch(
        (start_x + i * (card_width + gap), 1.2), card_width, 3.6,
        boxstyle="round,pad=0.15", facecolor='#F8F9FA', edgecolor='#E5E7EB',
        linewidth=1.2
    )
    ax.add_patch(rect)

    ax.text(cx, cy + 0.8, num, ha='center', va='center',
            fontsize=36, fontweight='bold', color=color)
    ax.text(cx, cy - 0.2, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color=TEXT)
    ax.text(cx, cy - 1.0, desc, ha='center', va='center',
            fontsize=7.5, color='#666666', linespacing=1.4)

ax.text(6, 5.5, 'Mister Maki Brand Awareness Metrics',
        ha='center', va='center', fontsize=14, fontweight='bold', color=TEXT)

fig.savefig(f'{OUT}/awareness_stats.png')
plt.close()

# ========================================================================
# 3. CONVERSION FUNNEL - conversion_funnel.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(6, 8.5, 'Awareness-to-Trial Conversion Funnel',
        ha='center', va='center', fontsize=14, fontweight='bold', color=TEXT)

awareness_w = 9.0
trial_w = awareness_w * (49 / 81)
funnel_cx = 6.0

aw_left = funnel_cx - awareness_w / 2
aw_right = funnel_cx + awareness_w / 2
tr_left = funnel_cx - trial_w / 2
tr_right = funnel_cx + trial_w / 2

trap_y_top = 7.5
trap_y_mid = 5.8
trap_y_bot = 4.1

trap_top = plt.Polygon(
    [[aw_left, trap_y_top], [aw_right, trap_y_top],
     [aw_right, trap_y_mid + 0.05], [aw_left, trap_y_mid + 0.05]],
    facecolor=STUDENTS, alpha=0.9, edgecolor='none'
)
ax.add_patch(trap_top)
ax.text(funnel_cx, (trap_y_top + trap_y_mid) / 2 + 0.05, '81%  Awareness',
        ha='center', va='center', fontsize=14, fontweight='bold', color=BG)
ax.text(funnel_cx + awareness_w / 2 + 0.3, (trap_y_top + trap_y_mid) / 2 + 0.05,
        '30 / 37', ha='left', va='center', fontsize=8, color='#888888')

gap_y = (trap_y_mid + trap_y_bot) / 2
ax.annotate('32 pp gap', xy=(funnel_cx + 2.5, gap_y),
            fontsize=10, fontweight='bold', color='#DC2626', ha='center', va='center')
ax.annotate('', xy=(funnel_cx + 1.2, gap_y - 0.25),
            xytext=(funnel_cx + 1.2, gap_y + 0.25),
            arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.5))

neck_left = funnel_cx - trial_w / 2
neck_right = funnel_cx + trial_w / 2
trap_bot = plt.Polygon(
    [[aw_left, trap_y_mid - 0.05], [aw_right, trap_y_mid - 0.05],
     [neck_right, trap_y_bot], [neck_left, trap_y_bot]],
    facecolor=STUDENTS, alpha=0.5, edgecolor='none'
)
ax.add_patch(trap_bot)
ax.text(funnel_cx, (trap_y_mid + trap_y_bot) / 2 - 0.1, '49%  Trial',
        ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT)
ax.text(funnel_cx + (aw_right + neck_right) / 4 + 0.8,
        (trap_y_mid + trap_y_bot) / 2 - 0.1,
        '18 / 37', ha='left', va='center', fontsize=8, color='#888888')

seg_y = 2.2
bar_h = 0.8
student_w = 4.4 * (44 / 58.3)
resident_w = 4.4

ax.barh(seg_y + 0.6, student_w, height=bar_h, color=STUDENTS, left=1.5)
ax.text(1.5 + student_w + 0.15, seg_y + 0.6, '44%', va='center', fontsize=10, fontweight='bold', color=STUDENTS)
ax.text(1.4, seg_y + 0.6, 'Students', va='center', ha='right', fontsize=9, color=TEXT)

ax.barh(seg_y - 0.6, resident_w, height=bar_h, color=RESIDENTS, left=1.5)
ax.text(1.5 + resident_w + 0.15, seg_y - 0.6, '58.3%', va='center', fontsize=10, fontweight='bold', color=RESIDENTS)
ax.text(1.4, seg_y - 0.6, 'Residents', va='center', ha='right', fontsize=9, color=TEXT)

ax.text(6, 3.3, 'Conversion Rate by Segment', ha='center', va='center',
        fontsize=10, fontweight='bold', color=TEXT)

fig.savefig(f'{OUT}/conversion_funnel.png')
plt.close()

# ========================================================================
# 4. TEMAKI FAMILIARITY - temaki_familiarity.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

labels = [
    '5 - Eat regularly',
    '4 - Have tried',
    '3 - Moderately familiar',
    '2 - Slightly familiar',
    '1 - Not at all familiar'
]
counts = [7, 15, 7, 6, 2]
pcts = [c / 37 * 100 for c in counts]

gradient = ['#0D6E6E', '#1A8A8A', '#2BA5A5', '#5DC0C0', '#94D8D8']

bars = ax.barh(range(len(labels)), pcts, color=gradient, height=0.65, edgecolor='none')

for bar, pct, count in zip(bars, pcts, counts):
    w = bar.get_width()
    if w > 8:
        ax.text(w - 1, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%  (n={count})', ha='right', va='center',
                fontsize=9, fontweight='bold', color=BG)
    else:
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%  (n={count})', ha='left', va='center',
                fontsize=9, fontweight='bold', color=TEXT)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 50)
ax.set_xlabel('')
ax.xaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)

ax.set_title('Temaki Familiarity Among Respondents', fontsize=14, fontweight='bold',
             color=TEXT, pad=20)
ax.text(0, -0.9, 'n = 37 respondents', fontsize=8, color='#888888',
        transform=ax.transData)

fig.savefig(f'{OUT}/temaki_familiarity.png')
plt.close()

# ========================================================================
# 5. VALUE PERCEPTION - value_perception.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

categories = ['Excellent Value', 'Somewhat Good', 'About Average', 'Somewhat Poor', 'Very Poor']
cust_pcts = [22.2, 5.6, 61.1, 11.1, 0.0]
noncust_pcts = [0.0, 21.1, 57.9, 21.1, 0.0]

y = np.arange(len(categories))
bar_h = 0.35

bars1 = ax.barh(y + bar_h / 2 + 0.02, cust_pcts, height=bar_h, color=STUDENTS,
                edgecolor='none', label='Customers (n=18)')
bars2 = ax.barh(y - bar_h / 2 - 0.02, noncust_pcts, height=bar_h, color=RESIDENTS,
                edgecolor='none', label='Non-Customers (n=19)')

for bar, pct in zip(bars1, cust_pcts):
    if pct > 0:
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', ha='left', va='center', fontsize=8, color=STUDENTS, fontweight='bold')
for bar, pct in zip(bars2, noncust_pcts):
    if pct > 0:
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', ha='left', va='center', fontsize=8, color=RESIDENTS, fontweight='bold')

highlight_idx = categories.index('Somewhat Poor')
rect = FancyBboxPatch(
    (-2, highlight_idx - 0.48), 78, 0.96,
    boxstyle="round,pad=0.05", facecolor='#FEF3C7', edgecolor='#F5A623',
    linewidth=1, alpha=0.3, zorder=0
)
ax.add_patch(rect)
ax.text(72, highlight_idx, '2x gap',
        ha='center', va='center', fontsize=8, fontweight='bold', color='#DC2626',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor='#DC2626', linewidth=0.8))

ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=9)
ax.set_xlim(0, 75)
ax.xaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)

legend = ax.legend(loc='lower right', frameon=False, fontsize=8)

ax.set_title('Value Perception: Customers vs Non-Customers\n$12\u201318 for 2\u20133 hand rolls',
             fontsize=14, fontweight='bold', color=TEXT, pad=20,
             fontstyle='normal')
ax.title.set_fontsize(14)
title_obj = ax.title
title_obj.set_fontweight('bold')
fig.text(0.5, 0.93, 'Value Perception: Customers vs Non-Customers',
         ha='center', fontsize=14, fontweight='bold', color=TEXT)
fig.text(0.5, 0.895, '$12\u201318 for 2\u20133 hand rolls',
         ha='center', fontsize=9, color='#666666', style='italic')
ax.set_title('', pad=30)

fig.savefig(f'{OUT}/value_perception.png')
plt.close()

# ========================================================================
# 6. SATISFACTION BARS - satisfaction_bars.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

dims = [
    'Portion Size vs Price',
    'Ease of Assembly (delivery)',
    'Packaging & Presentation',
    'Food Quality & Freshness',
    'Speed of Service',
]
scores = [4.11, 4.56, 4.72, 4.78, 4.83]
bar_colors = [AMBER, STUDENTS, STUDENTS, STUDENTS, STUDENTS]

y = np.arange(len(dims))
bars = ax.barh(y, scores, color=bar_colors, height=0.55, edgecolor='none')

for bar, score in zip(bars, scores):
    ax.text(bar.get_width() - 0.08, bar.get_y() + bar.get_height() / 2,
            f'{score:.2f}', ha='right', va='center',
            fontsize=10, fontweight='bold', color=BG)

ax.set_yticks(y)
ax.set_yticklabels(dims, fontsize=9)
ax.set_xlim(1, 5.3)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=8, color='#AAAAAA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['bottom'].set_color('#DDDDDD')
ax.spines['left'].set_visible(False)
ax.tick_params(left=False, bottom=False)

ax.axvline(x=4.5, color='#DDDDDD', linestyle='--', linewidth=0.8, alpha=0.5)

callout = FancyBboxPatch(
    (3.65, -1.25), 1.6, 0.85,
    boxstyle="round,pad=0.12", facecolor=TEAL, edgecolor='none', alpha=0.15,
    zorder=0
)
ax.add_patch(callout)
ax.text(4.45, -0.62, 'Reorder Intent', ha='center', va='center',
        fontsize=8, fontweight='bold', color=TEAL)
ax.text(4.45, -0.95, '4.44 / 5', ha='center', va='center',
        fontsize=13, fontweight='bold', color=TEAL)

ax.set_title('Customer Satisfaction (n = 18)', fontsize=14, fontweight='bold',
             color=TEXT, pad=20)

fig.savefig(f'{OUT}/satisfaction_bars.png')
plt.close()

# ========================================================================
# 7. SYNTHESIS FLOW - synthesis_flow.png
# ========================================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')

ax.text(8, 6.5, 'The Customer Journey: Where the Funnel Leaks',
        ha='center', va='center', fontsize=14, fontweight='bold', color=TEXT)

boxes = [
    {'x': 0.5, 'w': 2.6, 'color': STUDENTS, 'label': 'Awareness', 'big': '81%', 'sub': None},
    {'x': 3.9, 'w': 3.2, 'color': AMBER, 'label': 'Barriers', 'big': None,
     'sub': 'Concept\nUnfamiliarity\n\nValue\nPerception'},
    {'x': 7.9, 'w': 2.4, 'color': '#7AB8E0', 'label': 'Trial', 'big': '49%', 'sub': None},
    {'x': 11.1, 'w': 2.4, 'color': TEAL, 'label': 'Satisfaction', 'big': '4.7+/5', 'sub': None},
    {'x': 14.3, 'w': 1.4, 'color': TEAL, 'label': 'Reorder', 'big': '4.44/5', 'sub': None},
]

box_y = 2.8
box_h = 2.8

for b in boxes:
    rect = FancyBboxPatch(
        (b['x'], box_y), b['w'], box_h,
        boxstyle="round,pad=0.2", facecolor=b['color'], edgecolor='none',
        alpha=0.15, zorder=0
    )
    ax.add_patch(rect)

    border = FancyBboxPatch(
        (b['x'], box_y), b['w'], box_h,
        boxstyle="round,pad=0.2", facecolor='none', edgecolor=b['color'],
        linewidth=1.5, alpha=0.5, zorder=1
    )
    ax.add_patch(border)

    cx = b['x'] + b['w'] / 2
    cy = box_y + box_h / 2

    ax.text(cx, box_y + box_h - 0.35, b['label'],
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=b['color'])

    if b['big']:
        ax.text(cx, cy - 0.15, b['big'],
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=b['color'])
    elif b['sub']:
        ax.text(cx, cy - 0.2, b['sub'],
                ha='center', va='center', fontsize=8, color=b['color'],
                linespacing=1.3)

arrow_color = '#CCCCCC'
arrow_pairs = [
    (boxes[0]['x'] + boxes[0]['w'] + 0.08, boxes[1]['x'] - 0.08),
    (boxes[1]['x'] + boxes[1]['w'] + 0.08, boxes[2]['x'] - 0.08),
    (boxes[2]['x'] + boxes[2]['w'] + 0.08, boxes[3]['x'] - 0.08),
    (boxes[3]['x'] + boxes[3]['w'] + 0.08, boxes[4]['x'] - 0.08),
]

arrow_y = box_y + box_h / 2
for x1, x2 in arrow_pairs:
    ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5,
                                connectionstyle='arc3,rad=0'))

ax.annotate('LEAK', xy=(5.5, box_y - 0.15),
            fontsize=8, fontweight='bold', color='#DC2626', ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEE2E2',
                      edgecolor='#DC2626', linewidth=0.8))
ax.annotate('', xy=(5.5, box_y + 0.05), xytext=(5.5, box_y - 0.05),
            arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5))

ax.annotate('STRENGTH', xy=(12.3, box_y - 0.15),
            fontsize=7, fontweight='bold', color=TEAL, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#D1FAE5',
                      edgecolor=TEAL, linewidth=0.8))

fig.savefig(f'{OUT}/synthesis_flow.png')
plt.close()

print('All 7 charts generated.')
