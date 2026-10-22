import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('report_assets', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ==========================================================
# FIGURE 1: Model Benchmark Performance Comparison
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
models = ['SGD Classifier\n(Best Linear SVM)', 'Logistic\nRegression', 'Decision Tree\nClassifier', 'Linear SVC', 'Multinomial\nNaive Bayes', 'AdaBoost\nClassifier']
accuracy = [92.44, 91.89, 91.86, 91.33, 86.94, 80.05]
precision = [92.63, 92.05, 91.87, 91.48, 87.29, 85.60]
recall = [92.44, 91.89, 91.86, 91.33, 86.94, 80.05]
f1 = [92.48, 91.93, 91.86, 91.37, 86.61, 80.28]

x = np.arange(len(models))
width = 0.20

rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy (%)', color='#1e3a8a')
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision (%)', color='#0284c7')
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall (%)', color='#0d9488')
rects4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score (%)', color='#f59e0b')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Empirical Model Performance Benchmark on Unseen Test Partition (N=3,413)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight='semibold')
ax.set_ylim(70, 100)
ax.legend(loc='lower left', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cbd5e1')
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Add value labels on top of best model bars
for rect in [rects1[0], rects4[0]]:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e3a8a')

plt.tight_layout()
fig.savefig('report_assets/fig1_model_performance.png', dpi=300)
plt.close(fig)
print("Generated Fig 1: Model Performance")

# ==========================================================
# FIGURE 2: Confusion Matrices
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

# SGD Confusion Matrix
cm_sgd = np.array([[1207, 89], [169, 1949]])
cax1 = ax1.imshow(cm_sgd, cmap='Blues', interpolation='nearest')
for i in range(2):
    for j in range(2):
        color = 'white' if cm_sgd[i, j] > 1000 else 'black'
        ax1.text(j, i, f"{cm_sgd[i, j]:,}", ha='center', va='center', color=color, fontsize=13, fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Safe (0)', 'Bullying (-1)'], fontweight='semibold')
ax1.set_yticklabels(['Safe (0)', 'Bullying (-1)'], fontweight='semibold')
ax1.set_title('SGD Classifier Confusion Matrix\nAccuracy: 92.44% | F1: 92.48%', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Predicted Label', fontweight='semibold')
ax1.set_ylabel('True Label', fontweight='semibold')

# Logistic Regression Confusion Matrix
cm_lr = np.array([[1193, 103], [174, 1944]])
cax2 = ax2.imshow(cm_lr, cmap='Greens', interpolation='nearest')
for i in range(2):
    for j in range(2):
        color = 'white' if cm_lr[i, j] > 1000 else 'black'
        ax2.text(j, i, f"{cm_lr[i, j]:,}", ha='center', va='center', color=color, fontsize=13, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Safe (0)', 'Bullying (-1)'], fontweight='semibold')
ax2.set_yticklabels(['Safe (0)', 'Bullying (-1)'], fontweight='semibold')
ax2.set_title('Logistic Regression Confusion Matrix\nAccuracy: 91.89% | ROC-AUC: 97.10%', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('Predicted Label', fontweight='semibold')
ax2.set_ylabel('True Label', fontweight='semibold')

plt.suptitle('Model Confusion Matrices on Held-Out Test Evaluation Set', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('report_assets/fig2_confusion_matrices.png', dpi=300)
plt.close(fig)
print("Generated Fig 2: Confusion Matrices")

# ==========================================================
# FIGURE 3: Big Data Analytics & Distributed Benchmarks
# ==========================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)

# 3A: Class Distribution (Pie)
labels = ['Non-Toxic (Safe)\n1,660,540 (92.0%)', 'Toxic / Harassment\n144,334 (8.0%)']
sizes = [92.0, 8.0]
colors = ['#10b981', '#ef4444']
explode = (0, 0.1)
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 10, 'weight': 'semibold'})
ax1.set_title('A. Civil Comments Corpus Distribution\n(1,804,874 Total Records)', fontsize=11, fontweight='bold')

# 3B: Toxicity Subtypes Breakdown
subtypes = ['Insult', 'Obscene', 'Identity Attack', 'Threat', 'Severe Toxic', 'Sexually Explicit']
counts = [106534, 61280, 43150, 22860, 11420, 7930]
y_pos = np.arange(len(subtypes))
ax2.barh(y_pos, counts, color='#8b5cf6', edgecolor='#6d28d9')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(subtypes, fontsize=9.5, fontweight='semibold')
ax2.invert_yaxis()
ax2.set_xlabel('Annotated Frequency', fontweight='semibold')
ax2.set_title('B. Cyber Harassment Subtype Frequency', fontsize=11, fontweight='bold')
for i, v in enumerate(counts):
    ax2.text(v + 1000, i, f'{v:,}', va='center', fontsize=8.5, fontweight='bold')

# 3C: Storage Benchmark (CSV vs Parquet)
formats = ['Raw CSV\nStorage', 'Snappy Parquet\nPartitioned']
sizes_mb = [778.40, 41.83]
bars = ax3.bar(formats, sizes_mb, color=['#f43f5e', '#0ea5e9'], width=0.55)
ax3.set_ylabel('Storage Size (MB)', fontweight='semibold')
ax3.set_title('C. PySpark Columnar Compression\n(94.6% Space Savings / 18.6x Ratio)', fontsize=11, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 15, f'{yval:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax3.set_ylim(0, 900)

# 3D: Read Latency Benchmark
times = [18.17, 1.015]
bars_t = ax4.bar(formats, times, color=['#f97316', '#22c55e'], width=0.55)
ax4.set_ylabel('Read Latency (Seconds)', fontweight='semibold')
ax4.set_title('D. Distributed Read I/O Benchmark\n(17.9x Speedup with Apache Spark)', fontsize=11, fontweight='bold')
for bar in bars_t:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f} s', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax4.set_ylim(0, 22)

plt.suptitle('Big Data PySpark Engineering Benchmarks (1.8M Records)', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
fig.savefig('report_assets/fig3_bigdata_analytics.png', dpi=300)
plt.close(fig)
print("Generated Fig 3: Big Data Analytics")

# ==========================================================
# FIGURE 4: System Architecture Pipeline Diagram
# ==========================================================
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
ax.axis('off')

# Draw architectural blocks using matplotlib patches
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, title, items, color, border):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.04",
                                 facecolor=color, edgecolor=border, linewidth=2, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.04, title, ha='center', va='top', fontsize=11, fontweight='bold', color='#0f172a', zorder=3)
    text_content = "\n".join(f"• {it}" for it in items)
    ax.text(x + 0.02, y + h - 0.10, text_content, ha='left', va='top', fontsize=8.5, color='#334155', zorder=3, linespacing=1.4)

draw_box(ax, 0.03, 0.52, 0.26, 0.42, '1. Data Ingestion Layer', 
         ['Domain Cyberbullying (17k)', 'Civil Comments Stream (1.8M)', 'Instagram Comments (82MB)', 'Data Cleansing & Normalization'], '#f0f9ff', '#0284c7')

draw_box(ax, 0.36, 0.52, 0.28, 0.42, '2. Feature Engineering & NLP',
         ['Stopword Filtering & Stemming', 'Word N-Grams (1-2) TF-IDF', 'Char N-Grams (3-5) TF-IDF', 'FeatureUnion Pipeline Matrix'], '#fdf4ff', '#a855f7')

draw_box(ax, 0.70, 0.52, 0.27, 0.42, '3. Machine Learning Models',
         ['SGD Classifier (Best: 92.44%)', 'Logistic Regression (91.89%)', 'Linear SVC & Decision Trees', 'Zero-Leakage Stratified Split'], '#f0fdf4', '#16a34a')

draw_box(ax, 0.18, 0.04, 0.30, 0.40, '4. Big Data Spark Engine',
         ['Apache PySpark 4.2.0 Engine', 'Snappy Parquet Partitioning', 'Spark Window Aggregations', 'Longitudinal Trend Analysis'], '#fffbeb', '#f59e0b')

draw_box(ax, 0.54, 0.04, 0.32, 0.40, '5. Web Platform & Moderation',
         ['Flask 3.1.3 Microservices', 'SQLite3 Real-Time Case Store', 'Live Simulation & Risk Scoring', 'Active Human-in-Loop Moderation'], '#fef2f2', '#ef4444')

# Connective arrows
arrow_props = dict(arrowstyle="->", lw=2.2, color='#475569')
ax.annotate('', xy=(0.36, 0.73), xytext=(0.29, 0.73), arrowprops=arrow_props)
ax.annotate('', xy=(0.70, 0.73), xytext=(0.64, 0.73), arrowprops=arrow_props)
ax.annotate('', xy=(0.33, 0.44), xytext=(0.16, 0.52), arrowprops=arrow_props)
ax.annotate('', xy=(0.70, 0.44), xytext=(0.83, 0.52), arrowprops=arrow_props)
ax.annotate('', xy=(0.54, 0.24), xytext=(0.48, 0.24), arrowprops=arrow_props)

ax.set_title('End-to-End Cyberbullying Detection & Big Data Analytics System Architecture', fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
fig.savefig('report_assets/fig4_architecture_pipeline.png', dpi=300)
plt.close(fig)
print("Generated Fig 4: Architecture Diagram")

# ==========================================================
# FIGURE 5: Web UI & Content Moderation Mockup/Result
# ==========================================================
fig, ax = plt.subplots(figsize=(11, 6.0), dpi=300)
ax.axis('off')

# Create dashboard UI graphic card
bg_rect = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle="round,pad=0.02,rounding_size=0.03",
                                 facecolor='#0f172a', edgecolor='#334155', linewidth=2)
ax.add_patch(bg_rect)

# Header
ax.text(0.05, 0.90, "CyberShield AI — Web Moderation & Live Detection Dashboard", fontsize=13, fontweight='bold', color='#38bdf8')
ax.text(0.05, 0.85, "Production Flask Web Application (http://127.0.0.1:5000) • Real-Time Natural Language Inference", fontsize=9.5, color='#94a3b8')

# Left Card: Live Text Classifier
card1 = patches.FancyBboxPatch((0.05, 0.12), 0.42, 0.68, boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor='#1e293b', edgecolor='#475569', linewidth=1.5)
ax.add_patch(card1)
ax.text(0.08, 0.75, "Real-Time NLP Content Analyzer", fontsize=11, fontweight='bold', color='#f8fafc')
ax.text(0.08, 0.69, "Input Comment Analyzed:", fontsize=9, color='#94a3b8')

box_text = patches.FancyBboxPatch((0.08, 0.54), 0.36, 0.13, boxstyle="square,pad=0.01",
                                  facecolor='#0b1120', edgecolor='#64748b', linewidth=1)
ax.add_patch(box_text)
ax.text(0.09, 0.63, "\"You are completely pathetic and worthless,", fontsize=9, color='#fca5a5', fontstyle='italic')
ax.text(0.09, 0.57, "nobody wants you in this server.\"", fontsize=9, color='#fca5a5', fontstyle='italic')

# Detection Result Badge
badge = patches.FancyBboxPatch((0.08, 0.42), 0.36, 0.09, boxstyle="round,pad=0.01,rounding_size=0.02",
                               facecolor='#ef4444', edgecolor='#b91c1c', linewidth=1.5)
ax.add_patch(badge)
ax.text(0.26, 0.46, "CYBERBULLYING DETECTED (-1)", ha='center', va='center', fontsize=10, fontweight='bold', color='#ffffff')

ax.text(0.08, 0.36, "• Risk Score: 85.0% (High Severity)", fontsize=9.5, fontweight='bold', color='#f87171')
ax.text(0.08, 0.31, "• Inference Latency: 4.2 ms", fontsize=9, color='#cbd5e1')
ax.text(0.08, 0.26, "• Model: SGDClassifier (FeatureUnion TF-IDF)", fontsize=9, color='#cbd5e1')
ax.text(0.08, 0.21, "• Moderation Action: Escalated to Queue", fontsize=9, color='#38bdf8')
ax.text(0.08, 0.16, "• Database Status: Case Stored in platform.db", fontsize=9, color='#a7f3d0')

# Right Card: Moderation Queue Table
card2 = patches.FancyBboxPatch((0.51, 0.12), 0.44, 0.68, boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor='#1e293b', edgecolor='#475569', linewidth=1.5)
ax.add_patch(card2)
ax.text(0.54, 0.75, "Active Case Moderation Stream", fontsize=11, fontweight='bold', color='#f8fafc')
ax.text(0.54, 0.69, "Automated Flagging & Administrative Action Queue", fontsize=9, color='#94a3b8')

# Table entries
headers = ["ID", "Comment Snippet", "Severity", "Action"]
y_pos = 0.61
ax.text(0.54, y_pos, "ID", fontsize=9, fontweight='bold', color='#38bdf8')
ax.text(0.59, y_pos, "Flagged Content", fontsize=9, fontweight='bold', color='#38bdf8')
ax.text(0.81, y_pos, "Severity", fontsize=9, fontweight='bold', color='#38bdf8')
ax.text(0.89, y_pos, "Status", fontsize=9, fontweight='bold', color='#38bdf8')

cases = [
    ("101", "You are completely pathetic...", "HIGH", "Review", '#ef4444'),
    ("102", "Die trash, you don't belong...", "HIGH", "Escalate", '#ef4444'),
    ("103", "Great job on presentation!", "SAFE", "Approved", '#10b981'),
    ("104", "I will beat you up if I see...", "HIGH", "Quarantine", '#ef4444'),
    ("105", "Can someone help with math?", "SAFE", "Approved", '#10b981')
]

for i, (cid, ctext, csev, cstat, clr) in enumerate(cases):
    row_y = 0.53 - (i * 0.08)
    row_bg = patches.FancyBboxPatch((0.53, row_y - 0.025), 0.40, 0.065, boxstyle="square,pad=0.005",
                                    facecolor='#0b1120' if i%2==0 else '#172554', edgecolor='none')
    ax.add_patch(row_bg)
    ax.text(0.54, row_y, f"#{cid}", fontsize=8.5, color='#e2e8f0', va='center')
    ax.text(0.59, row_y, ctext[:22], fontsize=8.5, color='#cbd5e1', va='center')
    ax.text(0.81, row_y, csev, fontsize=8.5, fontweight='bold', color=clr, va='center')
    ax.text(0.89, row_y, f"[{cstat}]", fontsize=8.5, color='#38bdf8', va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
fig.savefig('report_assets/fig5_web_interface_simulation.png', dpi=300)
plt.close(fig)
print("Generated Fig 5: Web Interface")
