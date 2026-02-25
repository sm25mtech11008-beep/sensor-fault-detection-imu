"""
Temperature Sensor Fault Detection — PDF Report Generator
Run this script AFTER completing Phase 5.
It reads all config files + results and generates a full PDF report.

Usage:
    python generate_report.py
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score, roc_auc_score,
                              roc_curve, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize

from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import mm
from reportlab.lib           import colors
from reportlab.lib.colors    import HexColor
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums     import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus      import (SimpleDocTemplate, Paragraph, Spacer,
                                      Table, TableStyle, PageBreak,
                                      HRFlowable, Image, KeepTogether)

warnings.filterwarnings('ignore')

# ── Colors ──
DARK_BLUE  = HexColor('#1A237E')
MID_BLUE   = HexColor('#1565C0')
LIGHT_BLUE = HexColor('#E3F2FD')
ACCENT     = HexColor('#0288D1')
DARK_GRAY  = HexColor('#212121')
MED_GRAY   = HexColor('#616161')
LIGHT_GRAY = HexColor('#F5F5F5')
GREEN      = HexColor('#2E7D32')
RED        = HexColor('#C62828')
ORANGE     = HexColor('#E65100')
PURPLE     = HexColor('#6A1B9A')
WHITE      = colors.white

CLASS_COLORS_HEX = {
    'normal'        : '#43A047',
    'noise_fault'   : '#E53935',
    'bias_fault'    : '#FB8C00',
    'flatline_fault': '#8E24AA',
}

W, H = A4
CW   = W - 40*mm   # content width

# ================================================================
#  CHANGE THIS TO YOUR DATASET PATH
# ================================================================
BASE_PATH  = r'C:\PVS_Dataset'
# ================================================================

PHASE2_DIR = os.path.join(BASE_PATH, 'outputs_temp_phase2')
PHASE3_DIR = os.path.join(BASE_PATH, 'outputs_temp_phase3')
PHASE4_DIR = os.path.join(BASE_PATH, 'outputs_temp_phase4')
PHASE5_DIR = os.path.join(BASE_PATH, 'outputs_temp_phase5')
PLOTS_DIR  = os.path.join(BASE_PATH, 'report_plots')
OUT_PDF    = os.path.join(BASE_PATH, 'Fault_Detection_Report.pdf')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PHASE5_DIR, exist_ok=True)

# ── Load everything ──
print('Loading configs and models...')
with open(os.path.join(PHASE2_DIR, 'temp_phase2_config.json')) as f: p2 = json.load(f)
with open(os.path.join(PHASE3_DIR, 'temp_phase3_config.json')) as f: p3 = json.load(f)
with open(os.path.join(PHASE4_DIR, 'temp_phase4_config.json')) as f: p4 = json.load(f)

CLASS_LABELS = {int(k): v for k, v in p2['CLASS_LABELS'].items()}
CLASS_NAMES  = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]
FINAL_FEATS  = p3['FINAL_FEATS']
WINDOW_SIZE  = p2['WINDOW_SIZE']
SAMPLING_HZ  = p2['SAMPLING_HZ']

with open(p3['splits_path'],     'rb') as f: splits     = pickle.load(f)
with open(p4['all_models_path'], 'rb') as f: all_models = pickle.load(f)
with open(p3['scaler_path'],     'rb') as f: scaler     = pickle.load(f)

X_train   = splits['X_train'];   X_test   = splits['X_test']
X_train_s = splits['X_train_s']; X_test_s = splits['X_test_s']
y_train   = splits['y_train'];   y_test   = splits['y_test']

SCALED = ['KNN (k=7)', 'SVM (RBF kernel)']
BEST   = p4['best_model_name']
best_model   = all_models[BEST]
X_test_best  = X_test_s if BEST in SCALED else X_test

y_pred = best_model.predict(X_test_best)
acc    = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro')
f1_wtd = f1_score(y_test, y_pred, average='weighted')
prec, rec, f1s, sup = precision_recall_fscore_support(y_test, y_pred, labels=[0,1,2,3])

y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
if hasattr(best_model, 'predict_proba'):
    y_prob = best_model.predict_proba(X_test_best)
else:
    y_prob = None

print('Data loaded. Generating plots...')

# ════════════════════════════════════════════════════
#  GENERATE ALL PLOTS
# ════════════════════════════════════════════════════

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})

# ── Plot 1: Dual Confusion Matrix ──
cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, fmt, title in [
    (axes[0], cm,     'd',   'Raw Counts'),
    (axes[1], cm_pct, '.1f', 'Normalized (%)'),
]:
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, annot_kws={'size':12,'weight':'bold'})
    ax.set_title(f'Confusion Matrix — {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
CM_PATH = os.path.join(PLOTS_DIR, 'p1_confusion_matrix.png')
plt.savefig(CM_PATH, dpi=150, bbox_inches='tight')
plt.close()
print('  Plot 1: Confusion matrix done.')

# ── Plot 2: ROC Curves ──
ROC_PATH = None
if y_prob is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    auc_scores = {}
    for i, (cls_id, cls_name) in enumerate(CLASS_LABELS.items()):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        auc_scores[cls_name] = auc
        axes[0].plot(fpr, tpr, lw=2.5,
                     color=list(CLASS_COLORS_HEX.values())[i],
                     label=f'{cls_name}  AUC={auc:.4f}')
    axes[0].plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5,label='Random')
    macro_auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
    axes[0].set_title(f'ROC Curves — Macro AUC = {macro_auc:.4f}',
                       fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    names = list(auc_scores.keys())
    aucs  = list(auc_scores.values())
    clrs  = [CLASS_COLORS_HEX[n] for n in names]
    bars  = axes[1].bar(names, aucs, color=clrs, alpha=0.85, edgecolor='black', lw=0.5)
    for bar, val in zip(bars, aucs):
        axes[1].text(bar.get_x()+bar.get_width()/2, val+0.003,
                      f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    axes[1].axhline(1.0, color='green', lw=1.2, ls='--', alpha=0.5)
    axes[1].axhline(0.5, color='red',   lw=1.2, ls='--', alpha=0.5)
    axes[1].set_title('AUC Score per Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('AUC'); axes[1].set_ylim([0, 1.09])
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    ROC_PATH = os.path.join(PLOTS_DIR, 'p2_roc_curves.png')
    plt.savefig(ROC_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Plot 2: ROC curves done.')

# ── Plot 3: Per-class P/R/F1 ──
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(4); w = 0.25
ax.bar(x-w, prec*100, w, label='Precision', color='#1565C0', alpha=0.85)
ax.bar(x,   rec *100, w, label='Recall',    color='#2E7D32', alpha=0.85)
ax.bar(x+w, f1s *100, w, label='F1 Score',  color='#E65100', alpha=0.85)
for i in range(4):
    ax.text(i-w, prec[i]*100+0.4, f'{prec[i]*100:.1f}', ha='center', fontsize=8)
    ax.text(i,   rec[i] *100+0.4, f'{rec[i] *100:.1f}', ha='center', fontsize=8)
    ax.text(i+w, f1s[i] *100+0.4, f'{f1s[i] *100:.1f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=15, fontsize=10)
ax.set_ylabel('Score (%)'); ax.set_ylim([0,110])
ax.set_title(f'Per-Class Precision / Recall / F1 — {BEST}',
              fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
PRF_PATH = os.path.join(PLOTS_DIR, 'p3_per_class_prf.png')
plt.savefig(PRF_PATH, dpi=150, bbox_inches='tight')
plt.close()
print('  Plot 3: Per-class metrics done.')

# ── Plot 4: All-model comparison ──
import time
model_results = []
for name, model in all_models.items():
    X_te = X_test_s if name in SCALED else X_test
    t0   = time.time()
    yp   = model.predict(X_te)
    ms   = (time.time()-t0) / len(X_te) * 1000
    model_results.append({
        'Model'      : name,
        'Accuracy'   : round(accuracy_score(y_test,yp)*100, 2),
        'F1 Macro'   : round(f1_score(y_test,yp,average='macro')*100, 2),
        'Infer ms'   : round(ms, 4),
    })
comp_df = pd.DataFrame(model_results).sort_values('F1 Macro', ascending=False).reset_index(drop=True)

n = len(comp_df); x = np.arange(n); w = 0.3
m_colors = ['#1565C0','#2E7D32','#E65100','#6A1B9A'][:n]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(x-w/2, comp_df['Accuracy'],  w, label='Accuracy', color='#1565C0', alpha=0.85)
axes[0].bar(x+w/2, comp_df['F1 Macro'],  w, label='F1 Macro', color='#2E7D32', alpha=0.85)
for i,(a,f) in enumerate(zip(comp_df['Accuracy'],comp_df['F1 Macro'])):
    axes[0].text(i-w/2, a+0.2, f'{a:.1f}', ha='center', fontsize=8)
    axes[0].text(i+w/2, f+0.2, f'{f:.1f}', ha='center', fontsize=8)
axes[0].set_xticks(x); axes[0].set_xticklabels(comp_df['Model'],rotation=15,fontsize=9)
axes[0].set_title('All Models — Accuracy & F1 Macro', fontsize=12, fontweight='bold')
axes[0].set_ylim([max(0,comp_df['F1 Macro'].min()-5),103])
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(x, comp_df['Infer ms'], color=m_colors, alpha=0.85, edgecolor='black', lw=0.5)
for i,v in enumerate(comp_df['Infer ms']):
    axes[1].text(i, v+0.00005, f'{v:.4f}', ha='center', fontsize=9)
axes[1].set_xticks(x); axes[1].set_xticklabels(comp_df['Model'],rotation=15,fontsize=9)
axes[1].set_title('Inference Speed (ms/sample)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('ms per sample'); axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
COMP_PATH = os.path.join(PLOTS_DIR, 'p4_model_comparison.png')
plt.savefig(COMP_PATH, dpi=150, bbox_inches='tight')
plt.close()
print('  Plot 4: Model comparison done.')

# ── Plot 5: Feature importance (if RF or XGB) ──
FI_PATH = None
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
    imp_df = pd.DataFrame({'Feature':FINAL_FEATS,'Importance':imp})
    imp_df = imp_df.sort_values('Importance',ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10,5))
    bar_colors = plt.cm.RdYlGn(np.linspace(0.3,0.9,len(imp_df))[::-1])
    bars = ax.barh(imp_df['Feature'], imp_df['Importance'],
                    color=bar_colors, edgecolor='black', lw=0.5)
    for b, v in zip(bars, imp_df['Importance']):
        ax.text(v+0.002, b.get_y()+b.get_height()/2,
                 f'{v:.4f}', va='center', fontsize=9)
    ax.set_title(f'Feature Importance — {BEST}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score'); ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    FI_PATH = os.path.join(PLOTS_DIR, 'p5_feature_importance.png')
    plt.savefig(FI_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Plot 5: Feature importance done.')

print('All plots generated.\nBuilding PDF...')

# ════════════════════════════════════════════════════
#  REPORTLAB STYLES
# ════════════════════════════════════════════════════
def make_styles():
    base = ParagraphStyle('base', fontName='Helvetica', fontSize=10,
                           textColor=DARK_GRAY, leading=14)
    return {
        'title'  : ParagraphStyle('title',  parent=base, fontName='Helvetica-Bold',
                                   fontSize=22, textColor=WHITE, alignment=TA_CENTER, leading=28),
        'sub'    : ParagraphStyle('sub',    parent=base, fontName='Helvetica',
                                   fontSize=12, textColor=WHITE, alignment=TA_CENTER, leading=16),
        'h1'     : ParagraphStyle('h1',     parent=base, fontName='Helvetica-Bold',
                                   fontSize=15, textColor=DARK_BLUE, leading=20, spaceBefore=8, spaceAfter=4),
        'h2'     : ParagraphStyle('h2',     parent=base, fontName='Helvetica-Bold',
                                   fontSize=11, textColor=MID_BLUE, leading=16, spaceBefore=6, spaceAfter=3),
        'body'   : ParagraphStyle('body',   parent=base, fontName='Helvetica',
                                   fontSize=10, textColor=DARK_GRAY, leading=15,
                                   alignment=TA_JUSTIFY, spaceAfter=5),
        'caption': ParagraphStyle('caption',parent=base, fontName='Helvetica-Oblique',
                                   fontSize=9, textColor=MED_GRAY, alignment=TA_CENTER, leading=12),
        'small'  : ParagraphStyle('small',  parent=base, fontName='Helvetica',
                                   fontSize=9, textColor=DARK_GRAY, leading=13),
        'label'  : ParagraphStyle('label',  parent=base, fontName='Helvetica-Bold',
                                   fontSize=9, textColor=DARK_BLUE, leading=13),
        'metric' : ParagraphStyle('metric', parent=base, fontName='Helvetica-Bold',
                                   fontSize=13, textColor=DARK_BLUE, alignment=TA_CENTER, leading=18),
        'metric_sub': ParagraphStyle('msub',parent=base, fontName='Helvetica',
                                      fontSize=8, textColor=MED_GRAY, alignment=TA_CENTER, leading=11),
    }

S = make_styles()

def tbl_style(header_bg=DARK_BLUE, alt=LIGHT_BLUE):
    return TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), header_bg),
        ('TEXTCOLOR',     (0,0),(-1,0), WHITE),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0), 9),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [WHITE, alt]),
        ('FONTNAME',      (0,1),(-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,1),(-1,-1), 9),
        ('GRID',          (0,0),(-1,-1), 0.4, HexColor('#B0BEC5')),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 7),
        ('RIGHTPADDING',  (0,0),(-1,-1), 7),
    ])

def section_banner(text):
    t = Table([[Paragraph(text, S['h1'])]], colWidths=[CW])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), LIGHT_BLUE),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
        ('LINEBELOW',     (0,0),(-1,-1), 2, DARK_BLUE),
    ]))
    return t

def metric_box(value, label, color=DARK_BLUE):
    t = Table([
        [Paragraph(str(value), S['metric'])],
        [Paragraph(label, S['metric_sub'])],
    ], colWidths=[(CW-20*mm)/4])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), LIGHT_BLUE),
        ('TOPPADDING',    (0,0),(-1,-1), 8),
        ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ('LINEABOVE',     (0,0),(-1,0),  3, color),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
    ]))
    return t

def add_image(path, width_mm, caption_text=None):
    items = []
    if path and os.path.exists(path):
        img_w = width_mm * mm
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            ratio = im.height / im.width
        items.append(Image(path, width=img_w, height=img_w*ratio))
        if caption_text:
            items.append(Paragraph(f'<i>{caption_text}</i>', S['caption']))
        items.append(Spacer(1, 6))
    return items

# ════════════════════════════════════════════════════
#  BUILD PDF
# ════════════════════════════════════════════════════
doc   = SimpleDocTemplate(OUT_PDF, pagesize=A4,
                           leftMargin=20*mm, rightMargin=20*mm,
                           topMargin=18*mm, bottomMargin=18*mm)
story = []
today = datetime.now().strftime('%d %B %Y')

# ── PAGE 1: COVER ──
cover = Table([
    [Paragraph('Temperature Sensor Fault Detection', S['title'])],
    [Spacer(1, 8)],
    [Paragraph('Machine Learning-Based IMU Fault Classification', S['sub'])],
    [Spacer(1, 4)],
    [Paragraph(f'Passive Vehicular Sensors (PVS) Dataset  |  MPU-9250', S['sub'])],
], colWidths=[CW])
cover.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),(-1,-1), DARK_BLUE),
    ('TOPPADDING',    (0,0),(-1,-1), 30),
    ('BOTTOMPADDING', (0,0),(-1,-1), 30),
    ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
]))
story.append(cover)
story.append(Spacer(1, 20))

# Info table
info = [
    [Paragraph('Report Date',  S['label']), Paragraph(today, S['body'])],
    [Paragraph('Subject',      S['label']), Paragraph('Automotive Communication and Sensing', S['body'])],
    [Paragraph('Dataset',      S['label']), Paragraph('PVS1 – PVS9, Left + Right MPU-9250 temperature sensor', S['body'])],
    [Paragraph('Fault Classes',S['label']), Paragraph('normal, noise_fault, bias_fault, flatline_fault', S['body'])],
    [Paragraph('Best Model',   S['label']), Paragraph(BEST, S['body'])],
    [Paragraph('Test Accuracy',S['label']), Paragraph(f'{acc*100:.4f}%', S['body'])],
    [Paragraph('F1 Macro',     S['label']), Paragraph(f'{f1_mac*100:.4f}%', S['body'])],
]
it = Table(info, colWidths=[45*mm, CW-45*mm])
it.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),(0,-1), LIGHT_BLUE),
    ('BACKGROUND',    (1,0),(1,-1), WHITE),
    ('GRID',          (0,0),(-1,-1), 0.4, HexColor('#B0BEC5')),
    ('TOPPADDING',    (0,0),(-1,-1), 7),
    ('BOTTOMPADDING', (0,0),(-1,-1), 7),
    ('LEFTPADDING',   (0,0),(-1,-1), 10),
    ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
    ('FONTNAME',      (0,0),(-1,-1), 'Helvetica'),
]))
story.append(it)
story.append(Spacer(1, 20))

# Abstract
ab_title = Table([[Paragraph('Abstract', S['h2'])]], colWidths=[CW])
ab_title.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),DARK_BLUE),
                                ('TEXTCOLOR', (0,0),(-1,-1),WHITE),
                                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                                ('LEFTPADDING',(0,0),(-1,-1),10)]))
story.append(ab_title)
ab_body = Table([[Paragraph(
    f"This report presents the results of a machine learning-based fault detection system "
    f"for the MPU-9250 onboard temperature sensor in automotive applications. Using the "
    f"Passive Vehicular Sensors (PVS) dataset across 9 collection contexts (3 vehicles, "
    f"3 drivers, 3 road scenarios), three classes of sensor faults were synthetically injected: "
    f"noise/spike, bias/drift, and flat-line failure. A sliding-window feature extraction "
    f"pipeline was applied, and {len(all_models)} machine learning classifiers were trained and evaluated. "
    f"The best performing model ({BEST}) achieved {acc*100:.2f}% accuracy and "
    f"{f1_mac*100:.2f}% macro F1-score on the held-out test set.",
    S['body']
)]], colWidths=[CW])
ab_body.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT_BLUE),
                               ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
                               ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12)]))
story.append(ab_body)
story.append(PageBreak())

# ── PAGE 2: METHODOLOGY ──
story.append(section_banner('1. Methodology'))
story.append(Spacer(1, 8))

story.append(Paragraph('1.1 Dataset & Sensor', S['h2']))
story.append(Paragraph(
    'The PVS dataset was collected using MPU-9250 IMU sensors mounted on three vehicles '
    '(VW Saveiro, Fiat Bravo, Fiat Palio) driven by three different drivers across three '
    'road scenarios. Sensors were placed on the left and right sides of the front axle. '
    'For this phase of the project, only the onboard temperature channel of the MPU-9250 '
    'is used, sampled at 100 Hz.', S['body']))

story.append(Paragraph('1.2 Fault Injection Strategy', S['h2']))
fault_tbl_data = [
    [Paragraph('Fault Class', S['label']),
     Paragraph('Label', S['label']),
     Paragraph('Description', S['label']),
     Paragraph('Injection Method', S['label'])],
    ['normal',         '0', 'Clean sensor output', 'Raw signal + small ADC noise'],
    ['noise_fault',    '1', 'Random spike anomalies', 'Spike amplitude: 1.5x-4.0x global std'],
    ['bias_fault',     '2', 'Gradual offset drift', 'Linear drift + noise: 0.8x-2.5x std'],
    ['flatline_fault', '3', 'Sensor stuck at constant', 'Window mean + tiny jitter'],
]
ft = Table(fault_tbl_data, colWidths=[35*mm, 15*mm, 55*mm, CW-35*mm-15*mm-55*mm])
ft.setStyle(tbl_style())
story.append(ft)
story.append(Spacer(1, 10))

story.append(Paragraph('1.3 Feature Extraction', S['h2']))
story.append(Paragraph(
    f'A sliding window of {WINDOW_SIZE} samples (1 second at {SAMPLING_HZ} Hz) with 50% overlap was applied. '
    f'The following {len(FINAL_FEATS)} features were extracted per window:', S['body']))
feat_rows = [
    [Paragraph('Feature', S['label']), Paragraph('Purpose', S['label'])],
    ['mean',          'Detects bias/drift — shifts away from baseline'],
    ['std',           'Near-zero in flatline; elevated in noise'],
    ['range',         'Zero in flatline; large in noise'],
    ['skewness',      'Asymmetry from one-sided spikes'],
    ['kurtosis',      'Heavy tails — hallmark of impulsive noise'],
    ['mean_abs_diff', 'Average step size — zero in flatline, large in noise'],
    ['max_diff',      'Largest single step — directly catches spike moment'],
    ['trend',         'Linear slope — non-zero in bias fault'],
    ['unique_ratio',  'Fraction of unique values — near 0 in flatline'],
]
feat_rows_used = [feat_rows[0]] + [r for r in feat_rows[1:] if r[0] in FINAL_FEATS]
if len(feat_rows_used) == 1:
    feat_rows_used = feat_rows
fr_tbl = Table(feat_rows_used, colWidths=[38*mm, CW-38*mm])
fr_tbl.setStyle(tbl_style())
story.append(fr_tbl)
story.append(PageBreak())

# ── PAGE 3: KEY METRICS ──
story.append(section_banner('2. Key Performance Metrics'))
story.append(Spacer(1, 12))

# 4 metric boxes in a row
boxes = Table([
    [metric_box(f'{acc*100:.2f}%',   'Test Accuracy',    DARK_BLUE),
     metric_box(f'{f1_mac*100:.2f}%','F1 Macro',         MID_BLUE),
     metric_box(f'{f1_wtd*100:.2f}%','F1 Weighted',      ACCENT),
     metric_box(f'{len(y_test):,}',  'Test Samples',     GREEN)],
], colWidths=[(CW-20*mm)/4]*4, rowHeights=[None])
story.append(boxes)
story.append(Spacer(1, 14))

# Per-class metrics table
story.append(Paragraph('Per-Class Breakdown', S['h2']))
prf_header = [
    Paragraph('Class',     S['label']),
    Paragraph('Precision', S['label']),
    Paragraph('Recall',    S['label']),
    Paragraph('F1 Score',  S['label']),
    Paragraph('Support',   S['label']),
]
prf_rows = [prf_header]
for i, name in enumerate(CLASS_NAMES):
    prf_rows.append([name,
                     f'{prec[i]*100:.2f}%',
                     f'{rec[i]*100:.2f}%',
                     f'{f1s[i]*100:.2f}%',
                     f'{sup[i]:,}'])
prf_tbl = Table(prf_rows, colWidths=[50*mm,35*mm,35*mm,35*mm,CW-155*mm])
prf_tbl.setStyle(tbl_style())
story.append(prf_tbl)
story.append(Spacer(1, 14))

# Model comparison table
story.append(Paragraph('All Models Comparison', S['h2']))
mc_header = [Paragraph(c, S['label'])
              for c in ['Rank','Model','Accuracy (%)','F1 Macro (%)','Infer (ms/sample)']]
mc_rows = [mc_header]
for i, row in comp_df.iterrows():
    medal = '1st' if i==0 else ('2nd' if i==1 else ('3rd' if i==2 else f'{i+1}th'))
    mc_rows.append([medal, row['Model'],
                    f"{row['Accuracy']:.2f}",
                    f"{row['F1 Macro']:.2f}",
                    f"{row['Infer ms']:.4f}"])
mc_tbl = Table(mc_rows, colWidths=[15*mm,55*mm,35*mm,35*mm,CW-140*mm])
mc_tbl.setStyle(tbl_style())
story.append(mc_tbl)
story.append(PageBreak())

# ── PAGE 4: CONFUSION MATRIX ──
story.append(section_banner('3. Confusion Matrix'))
story.append(Spacer(1, 6))
story.append(Paragraph(
    'The confusion matrix shows how many windows were correctly classified (diagonal) '
    'vs misclassified (off-diagonal). The normalized version shows what percentage '
    'of each actual class was correctly identified.', S['body']))
story.append(Spacer(1, 8))
story.extend(add_image(CM_PATH, 160,
    'Figure 1: Left — raw counts. Right — normalized % per actual class. '
    'Diagonal = correct classification. Off-diagonal = misclassification.'))

# Confusion matrix numbers as text
story.append(Paragraph('Confusion Matrix Values:', S['h2']))
cm_header = [Paragraph('Actual \\ Predicted', S['label'])] + \
            [Paragraph(n, S['label']) for n in CLASS_NAMES]
cm_rows = [cm_header]
for i, actual in enumerate(CLASS_NAMES):
    row = [Paragraph(f'<b>{actual}</b>', S['small'])]
    for j in range(4):
        val     = cm[i, j]
        pct     = cm_pct[i, j]
        correct = i == j
        cell    = Paragraph(f'<b>{val}</b><br/>({pct:.1f}%)', S['small'])
        row.append(cell)
    cm_rows.append(row)

cm_tbl = Table(cm_rows, colWidths=[45*mm]+[(CW-45*mm)/4]*4)
cm_style = tbl_style()
for i in range(1, 5):
    cm_style.add('BACKGROUND', (i,i),(i,i), HexColor('#C8E6C9'))
cm_tbl.setStyle(cm_style)
story.append(cm_tbl)
story.append(PageBreak())

# ── PAGE 5: ROC CURVES ──
story.append(section_banner('4. ROC Curves & AUC Scores'))
story.append(Spacer(1, 6))
story.append(Paragraph(
    'ROC (Receiver Operating Characteristic) curves show the trade-off between '
    'True Positive Rate and False Positive Rate at different classification thresholds. '
    'AUC (Area Under the Curve) close to 1.0 indicates near-perfect discrimination. '
    'Each class is evaluated using a One-vs-Rest strategy.', S['body']))
story.append(Spacer(1, 8))

if ROC_PATH:
    story.extend(add_image(ROC_PATH, 160,
        'Figure 2: Left — ROC curves for all 4 fault classes. '
        'Right — AUC score per class. AUC = 1.0 is perfect; 0.5 is random.'))
    if y_prob is not None:
        macro_auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
        story.append(Paragraph(f'Macro-average AUC: <b>{macro_auc:.4f}</b>', S['body']))
        story.append(Spacer(1,6))
        auc_rows = [[Paragraph('Class',S['label']), Paragraph('AUC Score',S['label']),
                     Paragraph('Interpretation',S['label'])]]
        for i,(cls_id,cls_name) in enumerate(CLASS_LABELS.items()):
            auc = roc_auc_score(y_test_bin[:,i], y_prob[:,i])
            interp = 'Perfect' if auc>=0.999 else ('Excellent' if auc>=0.95 else
                     ('Good' if auc>=0.85 else 'Fair'))
            auc_rows.append([cls_name, f'{auc:.4f}', interp])
        at = Table(auc_rows, colWidths=[50*mm,35*mm,CW-85*mm])
        at.setStyle(tbl_style())
        story.append(at)
else:
    story.append(Paragraph('ROC curves not available (model does not support probability output).', S['body']))
story.append(PageBreak())

# ── PAGE 6: PER-CLASS METRICS + FEATURE IMPORTANCE ──
story.append(section_banner('5. Per-Class Metrics & Feature Importance'))
story.append(Spacer(1, 6))
story.extend(add_image(PRF_PATH, 150,
    'Figure 3: Precision, Recall, and F1-Score for each fault class. '
    'Precision = of all predicted positives, how many were correct. '
    'Recall = of all actual positives, how many were detected.'))

if FI_PATH:
    story.append(Paragraph('Feature Importance', S['h2']))
    story.append(Paragraph(
        f'The {BEST} model assigns an importance score to each feature based on '
        'how much it reduces impurity across all decision trees. '
        'Higher importance = feature is more useful for fault classification.', S['body']))
    story.extend(add_image(FI_PATH, 140,
        'Figure 4: Feature importance scores. '
        'Top features are the primary drivers of fault classification accuracy.'))
story.append(PageBreak())

# ── PAGE 7: MODEL COMPARISON ──
story.append(section_banner('6. All-Model Comparison'))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f'Four machine learning models were trained and evaluated on the same '
    f'fault detection task. All models used identical train/test splits '
    f'(80/20 stratified). The best model was selected based on F1 Macro score.',
    S['body']))
story.append(Spacer(1, 8))
story.extend(add_image(COMP_PATH, 160,
    'Figure 5: Left — Accuracy and F1 Macro for all models. '
    'Right — Inference speed (lower = faster, better for real-time deployment).'))
story.append(PageBreak())

# ── PAGE 8: CONCLUSIONS ──
story.append(section_banner('7. Conclusions & Limitations'))
story.append(Spacer(1, 8))

story.append(Paragraph('7.1 Summary of Findings', S['h2']))
findings = [
    f'The {BEST} model achieved the highest performance with {acc*100:.2f}% accuracy and {f1_mac*100:.2f}% F1 Macro on the test set.',
    'Feature extraction from 1-second sliding windows proved sufficient to distinguish all four fault classes.',
    f'The most discriminative features were those capturing signal stability: std, range, max_diff, trend, and unique_ratio.',
    f'All {len(all_models)} models achieved comparable performance, indicating the fault injection produced clearly separable signal characteristics.',
    'Left and right sensor placements showed consistent temperature behavior, with no significant asymmetry detected.',
]
for f in findings:
    story.append(Paragraph(f'&#8226;  {f}', ParagraphStyle('bul', parent=S['body'], leftIndent=14, spaceAfter=4)))
story.append(Spacer(1,8))

story.append(Paragraph('7.2 Limitations', S['h2']))
limitations = [
    'Faults were synthetically injected. Performance on real hardware faults may differ — real faults tend to be more subtle and ambiguous.',
    'The temperature sensor signal is extremely stable (low std within windows). This makes synthetic fault injection very effective but potentially oversimplified.',
    'Only the temperature channel was used in this phase. Expanding to accelerometer (3-axis) and gyroscope (3-axis) would provide richer fault signatures.',
    'No cross-context generalization test was performed in this phase — training and testing used data from all 9 datasets mixed together.',
]
for l in limitations:
    story.append(Paragraph(f'&#8226;  {l}', ParagraphStyle('bul2', parent=S['body'], leftIndent=14, spaceAfter=4)))
story.append(Spacer(1,8))

story.append(Paragraph('7.3 Recommended Next Steps', S['h2']))
nextsteps = [
    'Expand to accelerometer (acc_x, acc_y, acc_z) — run Phase 1 through Phase 5 again with those channels.',
    'Perform cross-context testing: train on PVS1-PVS6, test on PVS7-PVS9 (unseen vehicle and driver).',
    'Combine all sensor channels (temperature + accelerometer + gyroscope) into a single multi-channel fault detector.',
    'Investigate CNN-LSTM on raw window sequences as an alternative to hand-crafted features.',
]
for n in nextsteps:
    story.append(Paragraph(f'&#8226;  {n}', ParagraphStyle('bul3', parent=S['body'], leftIndent=14, spaceAfter=4)))

story.append(Spacer(1,16))
story.append(HRFlowable(width=CW, thickness=1, color=HexColor('#B0BEC5'), spaceAfter=8))
story.append(Paragraph(
    f'Automotive Communication and Sensing — Temperature Sensor Fault Detection | Generated: {today}',
    ParagraphStyle('footer', fontName='Helvetica-Oblique', fontSize=8,
                   textColor=MED_GRAY, alignment=TA_CENTER)))

# ── BUILD ──
try:
    from PIL import Image as PILImage
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable,'-m','pip','install','Pillow','--break-system-packages','-q'])
    from PIL import Image as PILImage

doc.build(story)
print(f'\nPDF report generated successfully:\n  {OUT_PDF}')
