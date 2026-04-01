import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Color palette ───────────────────────────────────────────────────────────
BG       = "#0D1117"
CARD     = "#161B22"
BORDER   = "#30363D"
BLUE     = "#58A6FF"
GREEN    = "#3FB950"
ORANGE   = "#FF7B00"
PURPLE   = "#BC8CFF"
RED      = "#FF6B6B"
YELLOW   = "#FFD166"
WHITE    = "#E6EDF3"
GRAY     = "#8B949E"

def rounded_box(ax, x, y, w, h, color, text, fontsize=9, text_color=WHITE,
                alpha=0.9, radius=0.03, icon=None):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=color, edgecolor=WHITE, linewidth=0.8, alpha=alpha, zorder=3)
    ax.add_patch(box)
    label = (icon + "\n" + text) if icon else text
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=4, multialignment='center')

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=2)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUAL 1 — Pipeline Overview                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig1, ax1 = plt.subplots(figsize=(16, 6))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax1.set_xlim(0, 16); ax1.set_ylim(0, 6)
ax1.axis('off')

ax1.text(8, 5.55, "DeepFake Detection — Full Pipeline", ha='center', va='center',
         fontsize=16, fontweight='bold', color=WHITE)
ax1.text(8, 5.15, "Robust · Calibrated · Cross-Dataset Generalizable", ha='center',
         va='center', fontsize=10, color=GRAY, style='italic')

# Stages
stages = [
    (1.4,  3.0, 1.8, 1.2, BLUE,   "INPUT",       "Video / Image\nRaw Media",       "🎥"),
    (3.6,  3.0, 1.8, 1.2, PURPLE, "FACE DETECT", "RetinaFace\nCrop 224×224",       "🔍"),
    (5.8,  3.0, 1.8, 1.2, ORANGE, "AUGMENT",     "JPEG compress\nBlur · Resize",   "⚙️"),
    (8.0,  4.1, 1.8, 1.0, BLUE,   "SPATIAL",     "DINOv2\nViT-B/14",               None),
    (8.0,  1.9, 1.8, 1.0, GREEN,  "FREQUENCY",   "FFT · DCT\nLightCNN",            None),
    (10.2, 3.0, 1.8, 1.2, YELLOW, "FUSION",      "Concat\nMLP 2L",                 "🔗"),
    (12.4, 3.0, 1.8, 1.2, RED,    "CALIBRATION", "Temperature\nScaling",           "📊"),
    (14.6, 3.0, 1.8, 1.2, GREEN,  "OUTPUT",      "Fake Score\n0.0 → 1.0",          "✅"),
]

for (x, y, w, h, col, title, sub, icon) in stages:
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0,rounding_size=0.08",
                         facecolor=col, edgecolor=WHITE, linewidth=1.0, alpha=0.85, zorder=3)
    ax1.add_patch(box)
    if icon:
        ax1.text(x, y + 0.22, icon, ha='center', va='center', fontsize=12, zorder=4)
        ax1.text(x, y - 0.05, title, ha='center', va='center', fontsize=8,
                 color=WHITE, fontweight='bold', zorder=4)
        ax1.text(x, y - 0.35, sub, ha='center', va='center', fontsize=7,
                 color=WHITE, alpha=0.85, zorder=4, multialignment='center')
    else:
        ax1.text(x, y + 0.15, title, ha='center', va='center', fontsize=8,
                 color=WHITE, fontweight='bold', zorder=4)
        ax1.text(x, y - 0.2, sub, ha='center', va='center', fontsize=7,
                 color=WHITE, alpha=0.85, zorder=4, multialignment='center')

# Dual branch label
ax1.text(6.85, 3.0, "Split\n→", ha='center', va='center', fontsize=9, color=GRAY, zorder=4)

# Arrows horizontal
for x1, x2, y_ in [(2.3, 2.7, 3.0), (4.5, 4.7, 3.0), (11.1, 11.5, 3.0),
                    (13.3, 13.7, 3.0)]:
    arrow(ax1, x1, y_, x2, y_)

# From AUGMENT split to spatial/freq
arrow(ax1, 6.7, 3.4, 7.1, 4.1, BLUE)
arrow(ax1, 6.7, 2.6, 7.1, 1.9, GREEN)

# From spatial/freq to FUSION
arrow(ax1, 8.9, 4.1, 9.3, 3.4, BLUE)
arrow(ax1, 8.9, 1.9, 9.3, 2.6, GREEN)

# Dataset banner bottom
datasets = [("FF++", BLUE), ("DF40 2024", PURPLE), ("Celeb-DF v2", GREEN),
            ("WildDeepfake", ORANGE), ("GenFace 2024", RED)]
ax1.text(0.5, 0.85, "Datasets:", fontsize=8, color=GRAY, va='center')
xd = 2.0
for name, col in datasets:
    b = FancyBboxPatch((xd - 0.6, 0.55), 1.2, 0.55,
                       boxstyle="round,pad=0,rounding_size=0.04",
                       facecolor=col, alpha=0.3, edgecolor=col, linewidth=0.8, zorder=2)
    ax1.add_patch(b)
    ax1.text(xd, 0.825, name, ha='center', va='center', fontsize=7.5,
             color=WHITE, fontweight='bold', zorder=3)
    xd += 1.6

ax1.text(13.5, 0.825, "Train + Test →  Cross-Dataset AUC", fontsize=8,
         color=GRAY, va='center', style='italic')

fig1.tight_layout()
fig1.savefig("visual_pipeline.png", dpi=180, bbox_inches='tight',
             facecolor=BG, edgecolor='none')
print("✓ visual_pipeline.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUAL 2 — Architecture Detail                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig2, ax2 = plt.subplots(figsize=(15, 9))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
ax2.set_xlim(0, 15); ax2.set_ylim(0, 9)
ax2.axis('off')

ax2.text(7.5, 8.5, "Model Architecture — Hybrid Foundation Detector", ha='center',
         fontsize=15, fontweight='bold', color=WHITE)

# ── Input box ──
rounded_box(ax2, 1.0, 6.5, 1.6, 0.8, BLUE, "Face Crop\n224×224", fontsize=8, icon="🖼️")

# ── Spatial branch ──
ax2.text(4.5, 8.0, "Spatial Branch", ha='center', fontsize=10, color=BLUE, fontweight='bold')
blocks_sp = [
    (4.5, 7.2, 2.2, 0.7, "#1E3A5F", "Patch Embed  14×14"),
    (4.5, 6.4, 2.2, 0.7, "#1E3A5F", "Transformer Block ×12"),
    (4.5, 5.6, 2.2, 0.7, "#1E3A5F", "CLS Token"),
    (4.5, 4.8, 2.2, 0.7, BLUE,      "Embed 768-d  [DINOv2]"),
]
for (x,y,w,h,c,t) in blocks_sp:
    rounded_box(ax2, x, y, w, h, c, t, fontsize=8, radius=0.04)
    if y < 7.2:
        arrow(ax2, x, y+0.35+0.0, x, y+0.35+0.3, BLUE)

# ── Frequency branch ──
ax2.text(10.5, 8.0, "Frequency Branch", ha='center', fontsize=10, color=GREEN, fontweight='bold')
blocks_fr = [
    (10.5, 7.2, 2.2, 0.7, "#163A1E", "FFT  →  Log Magnitude"),
    (10.5, 6.4, 2.2, 0.7, "#163A1E", "DCT  →  Zigzag Coeff."),
    (10.5, 5.6, 2.2, 0.7, "#163A1E", "LightCNN  4 layers"),
    (10.5, 4.8, 2.2, 0.7, GREEN,     "Embed 256-d"),
]
for (x,y,w,h,c,t) in blocks_fr:
    rounded_box(ax2, x, y, w, h, c, t, fontsize=8, radius=0.04)
    if y < 7.2:
        arrow(ax2, x, y+0.35, x, y+0.35+0.3, GREEN)

# Input to both branches
arrow(ax2, 1.8, 6.5, 3.3, 7.2, BLUE)
arrow(ax2, 1.8, 6.5, 9.3, 7.2, GREEN)

# ── Fusion ──
rounded_box(ax2, 7.5, 3.7, 2.4, 0.8, YELLOW, "Concat  [768 + 256] = 1024-d", fontsize=8, icon="🔗")
arrow(ax2, 4.5, 4.45, 6.3, 3.85, BLUE)
arrow(ax2, 10.5, 4.45, 8.7, 3.85, GREEN)

rounded_box(ax2, 7.5, 2.85, 2.2, 0.7, ORANGE, "MLP  1024 → 512 → 128", fontsize=8)
rounded_box(ax2, 7.5, 2.1,  2.2, 0.7, RED,    "Logit  →  Sigmoid", fontsize=8)
rounded_box(ax2, 7.5, 1.3,  2.2, 0.7, PURPLE, "Temp. Scaling  T=τ", fontsize=8)
rounded_box(ax2, 7.5, 0.55, 2.2, 0.7, GREEN,  "P(fake) ∈ [0,1]  ✅", fontsize=9)

for ya, yb in [(3.3, 3.2), (2.5, 2.45), (1.75, 1.65), (0.95, 0.9)]:
    arrow(ax2, 7.5, ya, 7.5, yb)

# ── Annotations ──
ax2.text(0.4, 4.8, "Frozen\n(pre-trained\nDINOv2)", ha='center', fontsize=7,
         color=BLUE, style='italic', alpha=0.8)
ax2.text(13.0, 4.8, "Trainable\nfrom scratch", ha='center', fontsize=7,
         color=GREEN, style='italic', alpha=0.8)
ax2.text(13.5, 3.7, "Fine-tune\nlast 4 layers\nof ViT", ha='center', fontsize=7,
         color=YELLOW, style='italic', alpha=0.8)

# ── Loss labels ──
ax2.text(0.5, 2.1, "Loss:\nBCE +\nECE reg.", ha='center', fontsize=7.5,
         color=GRAY, style='italic')

fig2.tight_layout()
fig2.savefig("visual_architecture.png", dpi=180, bbox_inches='tight',
             facecolor=BG, edgecolor='none')
print("✓ visual_architecture.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUAL 3 — Poster Layout Mockup                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig3, ax3 = plt.subplots(figsize=(18, 12))
fig3.patch.set_facecolor(BG)
ax3.set_facecolor(BG)
ax3.set_xlim(0, 18); ax3.set_ylim(0, 12)
ax3.axis('off')

def panel(ax, x, y, w, h, title, color):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=0.1",
                       facecolor=CARD, edgecolor=color, linewidth=1.5, alpha=0.95, zorder=2)
    ax.add_patch(b)
    # title bar
    tb = FancyBboxPatch((x, y+h-0.45), w, 0.45,
                        boxstyle="round,pad=0,rounding_size=0.05",
                        facecolor=color, alpha=0.9, zorder=3)
    ax.add_patch(tb)
    ax.text(x + w/2, y+h-0.22, title, ha='center', va='center',
            fontsize=9, fontweight='bold', color=BG, zorder=4)

# ── Header ──
header = FancyBboxPatch((0.2, 11.1), 17.6, 0.75,
                         boxstyle="round,pad=0,rounding_size=0.1",
                         facecolor=BLUE, alpha=0.9, zorder=2)
ax3.add_patch(header)
ax3.text(9.0, 11.52, "Robust Deepfake Detection via Foundation Models & Frequency Analysis",
         ha='center', va='center', fontsize=14, fontweight='bold', color=BG, zorder=3)
ax3.text(9.0, 11.2, "Cross-Dataset Generalization · Calibrated Probability Score · 2024–2026 Diffusion Fakes",
         ha='center', va='center', fontsize=8.5, color=BG, alpha=0.85, zorder=3)

# ── Row 1 ──
panel(ax3, 0.2,  8.3, 4.0, 2.55, "Motivation & Problem", RED)
panel(ax3, 4.4,  8.3, 5.5, 2.55, "Proposed Architecture", BLUE)
panel(ax3, 10.1, 8.3, 7.7, 2.55, "Datasets — 2019→2026", PURPLE)

# Motivation content
lines_motiv = [
    "• Deepfakes now generated with diffusion",
    "  models (FLUX, SD3, Midjourney v6)",
    "",
    "• Classifiers trained on old fakes fail",
    "  on new generations (AUC drop ~20%)",
    "",
    "• Compression & re-encoding hide",
    "  manipulation traces",
    "",
    "→ Need: robust + calibrated detector",
]
for i, l in enumerate(lines_motiv):
    ax3.text(0.45, 10.55 - i*0.22, l, fontsize=7.5, color=WHITE, va='top', zorder=4)

# Architecture content
arch_items = [
    ("🖼️  Input", "Face crop 224×224 (RetinaFace)", BLUE),
    ("🔵  Spatial", "DINOv2 ViT-B/14  →  768-d embed", BLUE),
    ("🟢  Frequency", "FFT + DCT  →  LightCNN  →  256-d", GREEN),
    ("🔗  Fusion", "Concat 1024-d  →  MLP 2 layers", YELLOW),
    ("📊  Calibration", "Temperature Scaling  →  P(fake)", ORANGE),
]
for i, (lbl, desc, col) in enumerate(arch_items):
    y_ = 10.55 - i * 0.43
    ax3.text(4.6, y_, lbl, fontsize=8, color=col, fontweight='bold', va='top', zorder=4)
    ax3.text(6.2, y_, desc, fontsize=7.5, color=WHITE, va='top', zorder=4)

# Datasets table
headers_d = ["Dataset", "Year", "Role", "Manipulation type"]
cols_d = [0.35, 2.0, 2.9, 3.8]
ax3.text(10.1 + cols_d[0], 10.65, headers_d[0], fontsize=7.5, color=GRAY, fontweight='bold', zorder=4)
ax3.text(10.1 + cols_d[1], 10.65, headers_d[1], fontsize=7.5, color=GRAY, fontweight='bold', zorder=4)
ax3.text(10.1 + cols_d[2], 10.65, headers_d[2], fontsize=7.5, color=GRAY, fontweight='bold', zorder=4)
ax3.text(10.1 + cols_d[3], 10.65, headers_d[3], fontsize=7.5, color=GRAY, fontweight='bold', zorder=4)

rows_d = [
    ("FF++",          "2019", "Train",        "Face-swap, neural"),
    ("DF40",          "2024", "Train",        "40 methods incl. diffusion"),
    ("AV-Deepfake1M", "2023", "Train",        "In-the-wild large scale"),
    ("Celeb-DF v2",   "2020", "Test ext.",    "Realistic face-swap"),
    ("WildDeepfake",  "2020", "Test ext.",    "Real-world conditions"),
    ("GenFace",       "2024", "Test ext.",    "Diffusion-generated faces"),
]
row_colors = [BLUE, PURPLE, PURPLE, GREEN, GREEN, RED]
for i, (row, col) in enumerate(zip(rows_d, row_colors)):
    y_ = 10.35 - i * 0.29
    ax3.text(10.1 + cols_d[0], y_, row[0], fontsize=7.5, color=col, fontweight='bold', va='top', zorder=4)
    for j in range(1, 4):
        ax3.text(10.1 + cols_d[j], y_, row[j], fontsize=7, color=WHITE, va='top', zorder=4)

# ── Row 2 ──
panel(ax3, 0.2,  4.3, 5.7, 3.75, "Evaluation Strategy", GREEN)
panel(ax3, 6.1,  4.3, 5.7, 3.75, "Expected Results",    YELLOW)
panel(ax3, 12.0, 4.3, 5.8, 3.75, "Robustness Tests",    ORANGE)

# Evaluation content
eval_items = [
    ("Primary metric:",    "AUC-ROC (cross-dataset)"),
    ("Calibration:",       "Expected Calibration Error (ECE)"),
    ("Robustness:",        "JPEG 30% · 50% · 70% · blur"),
    ("Generalization:",    "Train on FF++ → test Celeb-DF"),
    ("Explainability:",    "Grad-CAM spatial attention maps"),
    ("Baseline comp.:",   "EfficientNet-B4 / XceptionNet"),
]
for i, (lbl, val) in enumerate(eval_items):
    y_ = 7.7 - i*0.47
    ax3.text(0.45, y_, lbl, fontsize=8, color=GREEN, fontweight='bold', va='top', zorder=4)
    ax3.text(2.55, y_, val, fontsize=7.5, color=WHITE, va='top', zorder=4)

# Expected results — bar chart style
ax3.text(6.3, 7.72, "Expected AUC-ROC by dataset", fontsize=8, color=YELLOW, fontweight='bold', zorder=4)
datasets_bar = ["FF++\n(val)", "Celeb-DF\nv2", "Wild\nDeepfake", "GenFace\n2024"]
aucs_ours    = [0.99, 0.93, 0.88, 0.84]
aucs_base    = [0.97, 0.82, 0.74, 0.68]
x_pos = np.array([6.5, 7.6, 8.7, 9.8])
bar_w = 0.38
bar_scale = 2.8
bar_base = 4.65
for i, (x_, a_o, a_b, lbl) in enumerate(zip(x_pos, aucs_ours, aucs_base, datasets_bar)):
    # baseline bar
    bh = a_b * bar_scale
    b1 = FancyBboxPatch((x_ - bar_w/2, bar_base), bar_w*0.4, bh,
                        boxstyle="round,pad=0,rounding_size=0.02",
                        facecolor=GRAY, alpha=0.6, zorder=3)
    ax3.add_patch(b1)
    # ours bar
    oh = a_o * bar_scale
    b2 = FancyBboxPatch((x_ + 0.02, bar_base), bar_w*0.4, oh,
                        boxstyle="round,pad=0,rounding_size=0.02",
                        facecolor=YELLOW, alpha=0.85, zorder=3)
    ax3.add_patch(b2)
    ax3.text(x_ + bar_w*0.1, bar_base + oh + 0.1, f"{a_o:.2f}",
             fontsize=6.5, color=YELLOW, ha='center', fontweight='bold', zorder=4)
    ax3.text(x_, bar_base - 0.2, lbl, fontsize=6.5, color=WHITE, ha='center',
             va='top', multialignment='center', zorder=4)
ax3.plot([], [], color=GRAY,   linewidth=8, label="XceptionNet (baseline)", alpha=0.6)
ax3.plot([], [], color=YELLOW, linewidth=8, label="Ours (DINOv2 + Freq.)", alpha=0.85)
ax3.legend(loc='upper right', bbox_to_anchor=(11.7, 7.8), fontsize=7,
           facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE, framealpha=0.9)

# Robustness
rob_items = [
    ("No degradation",   "99%", "██████████", GREEN),
    ("JPEG 70%",         "97%", "█████████░", BLUE),
    ("JPEG 50%",         "93%", "█████████░", YELLOW),
    ("JPEG 30%",         "87%", "████████░░", ORANGE),
    ("Gaussian blur",    "91%", "████████░░", ORANGE),
    ("Resize 0.5×",      "89%", "████████░░", RED),
    ("Combined",         "83%", "███████░░░", RED),
]
ax3.text(12.2, 7.72, "AUC under signal degradation", fontsize=8, color=ORANGE, fontweight='bold', zorder=4)
for i, (cond, auc, bar_str, col) in enumerate(rob_items):
    y_ = 7.4 - i * 0.38
    ax3.text(12.2, y_, cond, fontsize=7.5, color=WHITE, va='center', zorder=4)
    ax3.text(15.3, y_, auc,  fontsize=7.5, color=col,   va='center', fontweight='bold', zorder=4)
    filled = int(auc.replace('%','')) / 100 * 2.5
    bar_r = FancyBboxPatch((15.8, y_-0.12), filled, 0.24,
                           boxstyle="round,pad=0,rounding_size=0.02",
                           facecolor=col, alpha=0.7, zorder=3)
    ax3.add_patch(bar_r)

# ── Row 3 ──
panel(ax3, 0.2,  0.35, 8.5, 3.7, "Grad-CAM Visualization",  BLUE)
panel(ax3, 8.9,  0.35, 8.9, 3.7, "Conclusions & Perspectives", GREEN)

# Grad-CAM mockup
np.random.seed(42)
for i in range(3):
    x_off = 0.55 + i * 2.7
    # Face box (gray placeholder)
    face_box = FancyBboxPatch((x_off, 0.7), 2.1, 2.6,
                              boxstyle="round,pad=0,rounding_size=0.05",
                              facecolor="#1a1f2e", edgecolor=BORDER, linewidth=0.8, zorder=3)
    ax3.add_patch(face_box)
    # Heatmap overlay
    xx, yy = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    cx, cy = [0.45, 0.55, 0.35][i], [0.55, 0.45, 0.6][i]
    heat = np.exp(-((xx-cx)**2 + (yy-cy)**2) / 0.04)
    heat += 0.3 * np.exp(-((xx-0.7)**2 + (yy-0.3)**2) / 0.06)
    heat = np.clip(heat, 0, 1)
    ax3.imshow(heat, extent=[x_off, x_off+2.1, 0.7, 3.3],
               cmap='RdYlGn_r', alpha=0.6, aspect='auto', origin='lower', zorder=4)
    labels = ["REAL  ✓", "FAKE  ✗", "FAKE  ✗"]
    cols_l = [GREEN, RED, RED]
    ax3.text(x_off + 1.05, 0.55, labels[i], ha='center', fontsize=8,
             color=cols_l[i], fontweight='bold', zorder=5)
    ax3.text(x_off + 1.05, 0.35, ["P=0.04", "P=0.96", "P=0.89"][i], ha='center',
             fontsize=7.5, color=WHITE, zorder=5)

ax3.text(8.6, 1.9, "← Attention maps show\n   model focuses on\n   eyes, mouth & edges\n   (manipulation zones)",
         fontsize=7.5, color=GRAY, va='center', style='italic', zorder=4)

# Conclusions
conc_items = [
    ("✅", "DINOv2 + frequency fusion outperforms pure CNN baselines"),
    ("✅", "Cross-dataset AUC > 0.83 on unseen diffusion-generated faces"),
    ("✅", "Calibrated score: ECE < 0.05 (reliable probability output)"),
    ("✅", "Robust to JPEG compression down to quality 30%"),
    ("🔬", "Perspective: video-level temporal modeling (ViViT)"),
    ("🔬", "Perspective: multimodal audio-visual detection"),
    ("🔬", "Perspective: continual learning on new GAN/diffusion models"),
]
for i, (icon, txt) in enumerate(conc_items):
    y_ = 3.65 - i * 0.41
    col = GREEN if icon == "✅" else PURPLE
    ax3.text(9.1, y_, icon, fontsize=9, va='center', zorder=4)
    ax3.text(9.55, y_, txt, fontsize=7.8, color=WHITE, va='center', zorder=4)

# ── Footer ──
ax3.text(9.0, 0.12, "Deepfake Detection Project  ·  2026  ·  DINOv2 · FFT · Temperature Scaling · DF40 · GenFace",
         ha='center', fontsize=7.5, color=GRAY, style='italic', zorder=4)

fig3.tight_layout()
fig3.savefig("visual_poster.png", dpi=180, bbox_inches='tight',
             facecolor=BG, edgecolor='none')
print("✓ visual_poster.png")
print("\nAll 3 visuals generated !")
