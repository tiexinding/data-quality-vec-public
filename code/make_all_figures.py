"""
重写 F1-F4 · 修重叠 + 双语 (英/中) 版本.

重叠修复:
  F1 (a): 备注框移 lower-right, 箭头文本和数字分层
  F1 (b): 箭头文本放柱外, 不盖数字
  F1 (c): gap= 标注用短形式放柱内底部, 备注框移 lower-right
  F1 (d): 箭头文本移 (1.6, 0.4), 备注框放 lower-right
  F3    : Kendall W 放 subplot 标题里, 不在右侧
  F4    : Legal-BERT 行列高亮改用虚线框, 且 linewidth 1.2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json
from scipy.stats import spearmanr

DATA = Path("/home/dingdang-ws/wsl-projects/claudecode/NPM_v13_commit_260324/NPM_v13_complete/20_NPM/0-update/数据战略地图/s_bar_战略地图_工作区/ablation_results_20260424")
OUT = DATA / "figures"

MODELS = ["minilm", "bge_small", "bge_large", "legal_bert"]
MODEL_LABELS = ["MiniLM", "BGE-small", "BGE-large", "Legal-BERT"]
# Wong 2011 (Nature Methods) color-blind-safe palette · 打印友好 · 高对比明快
# 顺序: MiniLM (深蓝) / BGE-small (蓝绿) / BGE-large (天蓝) / Legal-BERT (朱红 橙红)
# 替代之前 #D55E00 (深红 打印不明快), 改用 #D55E00 朱红 + 其他三色全面更换
MODEL_COLORS = ["#0072B2", "#009E73", "#56B4E9", "#D55E00"]

# 自定义打印友好 cmap (避免 RdBu_r/YlOrRd 端点的深酒红 #67001F)
from matplotlib.colors import LinearSegmentedColormap
# diverging: Wong 蓝 → 白 → Wong 朱红 (用于 F4 Spearman ρ ∈ [-1, 1])
WONG_DIVERGING = LinearSegmentedColormap.from_list(
    "wong_div", ["#0072B2", "#56B4E9", "#FFFFFF", "#FFB97A", "#D55E00"]
)
# sequential: 白 → 浅橙 → Wong 朱红 (用于 F3 heatmap 正值数据)
WONG_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "wong_seq", ["#FFFFFF", "#FFE0B0", "#F0A050", "#D55E00", "#9E3B00"]
)
MODEL_MARKERS = ["o", "s", "^", "D"]
DOMAINS = ["Pile-CC", "Wikipedia (en)", "ArXiv", "Github", "PubMed Central", "FreeLaw", "StackExchange", "USPTO Backgrounds"]
DOMAIN_SHORT_EN = ["Pile-CC", "Wiki", "ArXiv", "Github", "PubMed", "FreeLaw", "Stack", "USPTO"]
DOMAIN_SHORT_CN = ["Pile-CC", "维基", "ArXiv", "Github", "PubMed", "法律", "问答", "专利"]

COMPONENTS = ["s_bar_con", "s_bar_num", "s_bar_rep", "s_bar_div"]
COMP_LABELS_EN = ["$\\bar{s}_{con}$ (concentration)", "$\\bar{s}_{num}$ (unique fraction)",
                  "$\\bar{s}_{rep}$ (repetition)", "$\\bar{s}_{div}$ (Vendi diversity)"]
COMP_LABELS_CN = ["$\\bar{s}_{con}$ (聚类清晰度)", "$\\bar{s}_{num}$ (有效量)",
                  "$\\bar{s}_{rep}$ (重复度)", "$\\bar{s}_{div}$ (Vendi 多样性)"]

# ------- Load data -------
sbar = {m: pd.read_csv(DATA / f"sbar_{m}.csv") for m in MODELS}
trunc_mid = pd.read_csv(DATA / "sbar_trunc_mid.csv")
trunc_tail = pd.read_csv(DATA / "sbar_trunc_tail.csv")
aniso_rvc = pd.read_csv(DATA / "anisotropy_raw_vs_centered.csv")
aniso_all = pd.read_csv(DATA / "anisotropy_results.csv")
kendall = json.load(open(DATA / "kendall_spearman_analysis.json"))

def get(df, domain, col):
    return df[df["subset"] == domain][col].values[0]
def get_aniso(df, model, domain, col):
    row = df[(df["model"] == model) & (df["domain"] == domain)]
    return row[col].values[0] if len(row) else np.nan


def configure_mpl(lang):
    """Set matplotlib params. 中文版 sans-serif 首选 Noto CJK + DejaVu fallback."""
    import matplotlib.font_manager as fm
    # 强制注册 Noto CJK (ttc 集合包含 SC/TC/JP/KR/HK)
    for ttc in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
        try:
            fm.fontManager.addfont(ttc)
        except Exception:
            pass

    if lang == "cn":
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    else:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif"]

    mpl.rcParams.update({
        "axes.unicode_minus": False,  # 正常显示负号
        "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "figure.dpi": 120, "savefig.dpi": 300,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })

def T(en, cn, lang):
    """Translate helper."""
    return cn if lang == "cn" else en


# ===================================================================
# F1 · I6 four-mechanism decomposition (2x2)
# ===================================================================
def make_f1(lang):
    configure_mpl(lang)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.5))

    # --------- (a) Truncation ---------
    ax = axs[0, 0]
    positions_en = ["head\n[0:2000]", "mid\n[1500:3500]", "tail\n[-2000:]"]
    positions_cn = ["头段\n[0:2000]", "中段\n[1500:3500]", "尾段\n[-2000:]"]
    fl_div = [get(sbar["minilm"], "FreeLaw", "s_bar_div"),
              get(trunc_mid, "FreeLaw", "s_bar_div"),
              get(trunc_tail, "FreeLaw", "s_bar_div")]
    arx_div = [get(sbar["minilm"], "ArXiv", "s_bar_div"),
               get(trunc_mid, "ArXiv", "s_bar_div"),
               get(trunc_tail, "ArXiv", "s_bar_div")]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, fl_div, w, label="FreeLaw", color="#D55E00", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, arx_div, w, label=T("ArXiv (baseline)", "ArXiv (对照)", lang),
           color="#56B4E9", edgecolor="black", linewidth=0.5)
    # Ratio at the bar top (tight, not overlapping with note box)
    ratio_mid = fl_div[1] / fl_div[0]
    ax.annotate(f"↑ {ratio_mid:.2f}×",
                xy=(1 - w/2, fl_div[1]), xytext=(1 - w/2 - 0.3, fl_div[1] + 0.006),
                ha="center", fontsize=9, fontweight="bold", color="#D55E00",
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(positions_cn if lang == "cn" else positions_en)
    ax.set_ylabel(T("$\\bar{s}_{div}$ (Vendi Score)", "$\\bar{s}_{div}$ (Vendi 得分)", lang))
    ax.set_title(T("(a) Truncation effect · head suppresses $\\bar{s}_{div}$",
                   "(a) 截断效应 · 头段抑制 $\\bar{s}_{div}$", lang),
                 loc="left", fontweight="bold")
    ax.set_ylim(0, max(fl_div + arx_div) * 1.40)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    # Note box at BOTTOM RIGHT (避开箭头 + 数字)
    ax.text(0.97, 0.02,
        T(f"FreeLaw: head→mid $\\uparrow${ratio_mid:.2f}×\nArXiv: change < 13%\n→ Head is boilerplate-heavy",
          f"FreeLaw: 头→中 $\\uparrow${ratio_mid:.2f}×\nArXiv: 变化 < 13%\n→ 头段模板密集", lang),
        transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#D4A574", linewidth=0.5))

    # --------- (b) s̄_con ---------
    ax = axs[0, 1]
    con_vals = [get(sbar[m], "FreeLaw", "s_bar_con") for m in MODELS]
    bars = ax.bar(MODEL_LABELS, con_vals, color=MODEL_COLORS, edgecolor="black", linewidth=0.5)
    bars[3].set_hatch("//")
    # Values on top of bars
    for i, v in enumerate(con_vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold" if i == 3 else "normal")
    # Arrow text at top-left (safe zone)
    ratio_b = con_vals[3] / np.mean(con_vals[:3])
    ax.annotate(
        T(f"{ratio_b:.1f}× amplification\n(legal specialist sees\nsubcategory structure)",
          f"{ratio_b:.1f}× 放大\n(法律专模看到子类)", lang),
        xy=(3, con_vals[3]), xytext=(0.02, 0.88), textcoords="axes fraction",
        fontsize=9, color="#D55E00", fontweight="bold", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.3, connectionstyle="arc3,rad=-0.2")
    )
    ax.set_ylabel(T("$\\bar{s}_{con}$ on FreeLaw", "FreeLaw 上的 $\\bar{s}_{con}$", lang))
    ax.set_title(T("(b) Sub-category recognition · Legal-BERT ≠ generic",
                   "(b) 子类识别 · 法律专模 ≠ 通用模型", lang),
                 loc="left", fontweight="bold")
    ax.set_ylim(0, 0.135)
    ax.grid(axis="y", alpha=0.3)

    # --------- (c) Raw vs Centered ---------
    ax = axs[1, 0]
    raw_vals = [get_aniso(aniso_rvc, m, "FreeLaw", "raw_top10") for m in MODELS]
    cen_vals = [get_aniso(aniso_rvc, m, "FreeLaw", "centered_top10") for m in MODELS]
    gap_vals = [r - c for r, c in zip(raw_vals, cen_vals)]
    x = np.arange(len(MODELS)); w = 0.35
    ax.bar(x - w/2, raw_vals, w, label=T("raw top-10", "原始 top-10", lang),
           color="#56B4E9", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, cen_vals, w, label=T("centered top-10", "中心化 top-10", lang),
           color="#F0E442", edgecolor="black", linewidth=0.5)
    # gap values on top of pair with smaller font to avoid overlap
    for i in range(len(MODELS)):
        highest = max(raw_vals[i], cen_vals[i])
        ax.text(i, highest + 0.04,
                T(f"gap={gap_vals[i]:.2f}", f"差={gap_vals[i]:.2f}", lang),
                ha="center", fontsize=8.5,
                color="#D55E00" if gap_vals[i] >= 0.4 else "gray",
                fontweight="bold" if gap_vals[i] >= 0.4 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel(T("SVD top-10 ratio", "SVD top-10 奇异值²占比", lang))
    ax.set_title(T("(c) BGE global-bias artifact · gap ≥ 0.50",
                   "(c) BGE 全局偏置虚影 · 差 ≥ 0.50", lang),
                 loc="left", fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    # Note box at BOTTOM RIGHT
    ax.text(0.97, 0.02,
        T("BGE: raw 0.71 → centered 0.20\n→ >70% of 'low-rank' is\n   embedding bias, not text",
          "BGE: 原始 0.71 → 中心化 0.20\n→ BGE 下 >70% 的 '低秩' 是\n   嵌入空间偏置而非文本性质", lang),
        transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#D4A574", linewidth=0.5))

    # --------- (d) Centered-only ---------
    ax = axs[1, 1]
    bars = ax.bar(MODEL_LABELS, cen_vals, color=MODEL_COLORS, edgecolor="black", linewidth=0.5)
    bars[3].set_hatch("//")
    for i, v in enumerate(cen_vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold" if i == 3 else "normal")
    ax.axhline(y=cen_vals[0], color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.02, cen_vals[0], T("MiniLM\nbaseline", "MiniLM\n基线", lang),
            fontsize=8, va="center", ha="left", color="gray")
    # Arrow to Legal-BERT with text at MID-LEFT (safe zone)
    ax.annotate(
        T("Only model still\nshowing genuine\nlow-rank", "唯一呈现\n真实低秩的模型", lang),
        xy=(3, cen_vals[3]), xytext=(0.5, 0.40), textcoords="axes fraction",
        fontsize=9, color="#D55E00", fontweight="bold", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.3, connectionstyle="arc3,rad=0.15")
    )
    ax.set_ylabel(T("SVD centered top-10 (bias-removed)",
                    "SVD 中心化 top-10 (已去偏置)", lang))
    ax.set_title(T("(d) After bias removal · Legal-BERT alone retains low-rank",
                   "(d) 去除偏置后 · 仅法律专模保留低秩", lang),
                 loc="left", fontweight="bold")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.3)
    # Note box at BOTTOM LEFT (避开 arrow, MiniLM baseline label 在右边)
    ax.text(0.03, 0.02,
        T("MiniLM/BGE centered ≈ 0.19-0.26\n(comparable; no real low-rank)\n\n"
          "Legal-BERT centered = 0.57\n→ real templating, weaker\n   than generics claim",
          "MiniLM/BGE 中心化 ≈ 0.19-0.26\n(相当 · 无真实低秩)\n\n"
          "法律专模中心化 = 0.57\n→ 真有模板化 · 但比\n   通用模型报告的弱", lang),
        transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#D4A574", linewidth=0.5))

    plt.tight_layout()

    fn = f"F1_I6_four_mechanism_decomposition{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F1 ({lang}) saved")


# ===================================================================
# F2 · Vendi-SVD scatter
# ===================================================================
def make_f2(lang):
    configure_mpl(lang)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    points = []
    for mi, m in enumerate(MODELS):
        for d in DOMAINS:
            s_div = sbar[m][sbar[m]["subset"] == d]["s_bar_div"].values[0]
            d_sl = d.replace(" (en)", "_en").replace(" Central", "_Central").replace(" Backgrounds", "_Backgrounds")
            r = aniso_all[(aniso_all["model"] == m) & (aniso_all["domain"] == d_sl) & (aniso_all["mode"] == "l2norm")]
            if len(r) == 0:
                continue
            points.append((r["top10_ratio"].values[0], s_div, m, d, mi))
    for mi, m in enumerate(MODELS):
        xs = [p[0] for p in points if p[2] == m]
        ys = [p[1] for p in points if p[2] == m]
        ax.scatter(xs, ys, color=MODEL_COLORS[mi], marker=MODEL_MARKERS[mi], s=110,
                   label=MODEL_LABELS[mi], edgecolors="black", linewidths=0.7, zorder=3)
        for p in points:
            if p[2] == m and p[3] == "FreeLaw":
                # Stagger the FreeLaw annotation vertically per model
                y_offset = 0.003 + 0.001 * mi
                x_offset = 0.02 if mi < 2 else -0.02
                ha = "left" if mi < 2 else "right"
                ax.annotate("FreeLaw", xy=(p[0], p[1]), xytext=(p[0]+x_offset, p[1]*1.3),
                            fontsize=7.5, color=MODEL_COLORS[mi], fontweight="bold", ha=ha)
    xs_all = np.array([p[0] for p in points]); ys_all = np.array([p[1] for p in points])
    valid = (ys_all > 0) & (xs_all > 0)
    coef = np.polyfit(np.log(xs_all[valid]), np.log(ys_all[valid]), 1)
    x_fit = np.linspace(xs_all.min(), xs_all.max(), 50)
    y_fit = np.exp(coef[1]) * x_fit ** coef[0]
    ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.7, linewidth=1.5,
            label=T(f"Power-law fit: $y \\propto x^{{{coef[0]:.2f}}}$",
                    f"幂律拟合: $y \\propto x^{{{coef[0]:.2f}}}$", lang))
    rho_all, p_all = spearmanr(xs_all, ys_all)
    ax.set_xlabel(T("SVD top-10 ratio (l2norm) · more isotropic $\\leftarrow$  $\\rightarrow$ more anisotropic",
                    "SVD top-10 比值 (l2归一化) · 更各向同性 $\\leftarrow$  $\\rightarrow$ 更各向异性", lang))
    ax.set_ylabel(T("$\\bar{s}_{div}$ (Vendi Score, log scale)",
                    "$\\bar{s}_{div}$ (Vendi 得分, 对数轴)", lang))
    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.text(0.02, 0.03,
        T(f"Spearman $\\rho = {rho_all:.3f}$ (p = {p_all:.4f}, n=32)\n"
          "→ Two metrics near-mathematically equivalent\n"
          "→ Vendi cross-model incomparability is a\n"
          "   direct consequence of embedding geometry",
          f"Spearman $\\rho = {rho_all:.3f}$ (p = {p_all:.4f}, n=32)\n"
          "→ 两测度数学上近乎等价\n"
          "→ Vendi 跨模型绝对值不可比是\n"
          "   嵌入空间几何的直接后果", lang),
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4F8", edgecolor="#0072B2"))
    plt.tight_layout()
    fn = f"F2_vendi_svd_triangulation{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F2 ({lang}) saved: ρ = {rho_all:.3f}")


# ===================================================================
# F3 · 4-component heatmap (Kendall W in title, colorbar clean)
# ===================================================================
def make_f3(lang):
    configure_mpl(lang)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    comp_lbls = COMP_LABELS_CN if lang == "cn" else COMP_LABELS_EN
    domain_shorts = DOMAIN_SHORT_CN if lang == "cn" else DOMAIN_SHORT_EN
    for ci, (comp, label) in enumerate(zip(COMPONENTS, comp_lbls)):
        ax = axs[ci // 2, ci % 2]
        mat = np.zeros((4, 8))
        for mi, m in enumerate(MODELS):
            for di, d in enumerate(DOMAINS):
                mat[mi, di] = sbar[m][sbar[m]["subset"] == d][comp].values[0]
        im = ax.imshow(mat, cmap=WONG_SEQUENTIAL, aspect="auto")
        ax.set_xticks(np.arange(8)); ax.set_xticklabels(domain_shorts, rotation=40, ha="right")
        ax.set_yticks(np.arange(4)); ax.set_yticklabels(MODEL_LABELS)
        # Kendall W IN TITLE (no external text)
        w_val = kendall["kendall_w"][comp]["W"]
        w_color = "#2E7D32" if w_val >= 0.7 else ("#F57C00" if w_val >= 0.5 else "#C62828")
        verdict = T("strong", "强一致", lang) if w_val >= 0.7 else (
                  T("moderate", "中一致", lang) if w_val >= 0.5 else T("weak", "弱一致", lang))
        ax.set_title(f"({'abcd'[ci]}) {label}\nKendall W = {w_val:.3f} ({verdict})",
                     loc="left", fontweight="bold", color=w_color)
        # Cell values
        for mi in range(4):
            for di in range(8):
                v = mat[mi, di]
                norm_v = (v - mat.min()) / (mat.max() - mat.min() + 1e-12)
                txt_color = "white" if norm_v > 0.55 else "black"
                ax.text(di, mi, f"{v:.3f}"[:5], ha="center", va="center",
                        fontsize=7.5, color=txt_color)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    plt.tight_layout()
    fn = f"F3_sbar_4comp_heatmap{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F3 ({lang}) saved")


# ===================================================================
# F4 · Kendall's W + 4×4 Spearman matrices (no overlapping rectangles)
# ===================================================================
def make_f4(lang):
    configure_mpl(lang)
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
    comp_lbls = COMP_LABELS_CN if lang == "cn" else COMP_LABELS_EN
    # Labels with Legal-BERT highlighted differently
    model_labels_mark = MODEL_LABELS.copy()
    for ci, (comp, label) in enumerate(zip(COMPONENTS, comp_lbls)):
        ax = axs[ci // 2, ci % 2]
        rho_mat = np.array(kendall["pairwise_spearman"][comp]["rho_matrix"])
        w_val = kendall["kendall_w"][comp]["W"]
        p_val = kendall["kendall_w"][comp]["p"]
        im = ax.imshow(rho_mat, cmap=WONG_DIVERGING, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(model_labels_mark, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(model_labels_mark)
        # Color the Legal-BERT tick labels red
        for t in ax.get_xticklabels():
            if "Legal-BERT" in t.get_text():
                t.set_color("#D55E00"); t.set_fontweight("bold")
        for t in ax.get_yticklabels():
            if "Legal-BERT" in t.get_text():
                t.set_color("#D55E00"); t.set_fontweight("bold")
        # Cell values
        for i in range(4):
            for j in range(4):
                v = rho_mat[i, j]
                txt_color = "white" if abs(v) > 0.55 else "black"
                fw = "bold" if (v < 0 or abs(v) > 0.9) else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10.5, color=txt_color, fontweight=fw)
        # Verdict
        if w_val >= 0.7: verdict_color = "#2E7D32"; verdict = T("Strong", "强", lang)
        elif w_val >= 0.5: verdict_color = "#F57C00"; verdict = T("Moderate", "中", lang)
        else: verdict_color = "#C62828"; verdict = T("Weak", "弱", lang)
        ax.set_title(f"({'abcd'[ci]}) {label}\nW = {w_val:.3f} ({verdict}, p = {p_val:.3f})",
                     loc="left", fontweight="bold", color=verdict_color)
    plt.tight_layout(rect=(0, 0, 0.92, 1))
    # Shared colorbar
    cbar_ax = fig.add_axes((0.94, 0.12, 0.018, 0.75))
    fig.colorbar(im, cax=cbar_ax, label=T("Spearman $\\rho$", "Spearman $\\rho$", lang))
    fn = f"F4_kendall_w_spearman_matrices{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F4 ({lang}) saved")


# ===================================================================
# Main-study figures (sentence-transformers ONLY, 3 models · for paper §3)
# ===================================================================
ST_MODELS = ["minilm", "bge_small", "bge_large"]
ST_MODEL_LABELS = ["MiniLM", "BGE-small", "BGE-large"]
ST_MODEL_COLORS = ["#0072B2", "#009E73", "#56B4E9"]
ST_MODEL_MARKERS = ["o", "s", "^"]


def make_f2_st(lang):
    """F2 main-study version · 24 points · sentence-transformers only."""
    configure_mpl(lang)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    points = []
    for mi, m in enumerate(ST_MODELS):
        for d in DOMAINS:
            s_div = sbar[m][sbar[m]["subset"] == d]["s_bar_div"].values[0]
            d_sl = d.replace(" (en)", "_en").replace(" Central", "_Central").replace(" Backgrounds", "_Backgrounds")
            r = aniso_all[(aniso_all["model"] == m) & (aniso_all["domain"] == d_sl) & (aniso_all["mode"] == "l2norm")]
            if len(r) == 0:
                continue
            points.append((r["top10_ratio"].values[0], s_div, m, d, mi))
    for mi, m in enumerate(ST_MODELS):
        xs = [p[0] for p in points if p[2] == m]
        ys = [p[1] for p in points if p[2] == m]
        ax.scatter(xs, ys, color=ST_MODEL_COLORS[mi], marker=ST_MODEL_MARKERS[mi], s=110,
                   label=ST_MODEL_LABELS[mi], edgecolors="black", linewidths=0.7, zorder=3)
        for p in points:
            if p[2] == m and p[3] == "FreeLaw":
                y_offset = 0.003 + 0.001 * mi
                x_offset = 0.02 if mi < 2 else -0.02
                ha = "left" if mi < 2 else "right"
                ax.annotate("FreeLaw", xy=(p[0], p[1]), xytext=(p[0]+x_offset, p[1]*1.3),
                            fontsize=7.5, color=ST_MODEL_COLORS[mi], fontweight="bold", ha=ha)
    xs_all = np.array([p[0] for p in points]); ys_all = np.array([p[1] for p in points])
    valid = (ys_all > 0) & (xs_all > 0)
    coef = np.polyfit(np.log(xs_all[valid]), np.log(ys_all[valid]), 1)
    x_fit = np.linspace(xs_all.min(), xs_all.max(), 50)
    y_fit = np.exp(coef[1]) * x_fit ** coef[0]
    ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.7, linewidth=1.5,
            label=T(f"Power-law fit: $y \\propto x^{{{coef[0]:.2f}}}$",
                    f"幂律拟合: $y \\propto x^{{{coef[0]:.2f}}}$", lang))
    rho_all, p_all = spearmanr(xs_all, ys_all)
    ax.set_xlabel(T("SVD top-10 ratio (l2norm) · more isotropic $\\leftarrow$  $\\rightarrow$ more anisotropic",
                    "SVD top-10 比值 (l2归一化) · 更各向同性 $\\leftarrow$  $\\rightarrow$ 更各向异性", lang))
    ax.set_ylabel(T("$\\bar{s}_{div}$ (Vendi Score, log scale)",
                    "$\\bar{s}_{div}$ (Vendi 得分, 对数轴)", lang))
    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.text(0.02, 0.03,
        T(f"Spearman $\\rho = {rho_all:.3f}$ (p = {p_all:.4g}, n=24)\n"
          "→ Two metrics near-mathematically equivalent\n"
          "→ Vendi cross-model incomparability is a\n"
          "   direct consequence of embedding geometry",
          f"Spearman $\\rho = {rho_all:.3f}$ (p = {p_all:.4g}, n=24)\n"
          "→ 两测度数学上近乎等价\n"
          "→ Vendi 跨模型绝对值不可比是\n"
          "   嵌入空间几何的直接后果", lang),
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4F8", edgecolor="#0072B2"))
    plt.tight_layout()
    fn = f"F2_vendi_svd_st_24pt{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F2-ST ({lang}) saved: ρ = {rho_all:.3f}, n=24")


def make_f3_st(lang):
    """F3 main-study version · 3 sentence-transformers models × 8 domains × 4 components."""
    configure_mpl(lang)
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    comp_lbls = COMP_LABELS_CN if lang == "cn" else COMP_LABELS_EN
    domain_shorts = DOMAIN_SHORT_CN if lang == "cn" else DOMAIN_SHORT_EN
    for ci, (comp, label) in enumerate(zip(COMPONENTS, comp_lbls)):
        ax = axs[ci // 2, ci % 2]
        mat = np.zeros((3, 8))
        for mi, m in enumerate(ST_MODELS):
            for di, d in enumerate(DOMAINS):
                mat[mi, di] = sbar[m][sbar[m]["subset"] == d][comp].values[0]
        im = ax.imshow(mat, cmap=WONG_SEQUENTIAL, aspect="auto")
        ax.set_xticks(np.arange(8)); ax.set_xticklabels(domain_shorts, rotation=40, ha="right")
        ax.set_yticks(np.arange(3)); ax.set_yticklabels(ST_MODEL_LABELS)
        # Compute Kendall W from the 3-model rank consistency
        from scipy.stats import rankdata
        ranks = np.array([rankdata(mat[i, :]) for i in range(3)])
        n_items = 8; n_raters = 3
        S = np.sum((np.sum(ranks, axis=0) - n_raters * (n_items + 1) / 2) ** 2)
        w_val = 12 * S / (n_raters ** 2 * (n_items ** 3 - n_items))
        w_color = "#2E7D32" if w_val >= 0.7 else ("#F57C00" if w_val >= 0.5 else "#C62828")
        verdict = T("strong", "强一致", lang) if w_val >= 0.7 else (
                  T("moderate", "中一致", lang) if w_val >= 0.5 else T("weak", "弱一致", lang))
        ax.set_title(f"({'abcd'[ci]}) {label}\nKendall W = {w_val:.3f} ({verdict})",
                     loc="left", fontweight="bold", color=w_color)
        for mi in range(3):
            for di in range(8):
                v = mat[mi, di]
                norm_v = (v - mat.min()) / (mat.max() - mat.min() + 1e-12)
                txt_color = "white" if norm_v > 0.55 else "black"
                ax.text(di, mi, f"{v:.3f}"[:5], ha="center", va="center",
                        fontsize=8, color=txt_color)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    plt.tight_layout()
    fn = f"F3_sbar_heatmap_st_3model{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F3-ST ({lang}) saved")


def make_f4_st(lang):
    """F4 main-study version · 3×3 pairwise Spearman matrices · sentence-transformers only."""
    configure_mpl(lang)
    fig, axs = plt.subplots(2, 2, figsize=(10, 9.5))
    comp_lbls = COMP_LABELS_CN if lang == "cn" else COMP_LABELS_EN
    from scipy.stats import rankdata
    for ci, (comp, label) in enumerate(zip(COMPONENTS, comp_lbls)):
        ax = axs[ci // 2, ci % 2]
        # Build 3x3 Spearman ρ matrix
        vals = np.zeros((3, 8))
        for mi, m in enumerate(ST_MODELS):
            for di, d in enumerate(DOMAINS):
                vals[mi, di] = sbar[m][sbar[m]["subset"] == d][comp].values[0]
        rho_mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    rho_mat[i, j] = 1.0
                else:
                    rho_mat[i, j] = spearmanr(vals[i, :], vals[j, :]).correlation
        # Kendall W
        ranks = np.array([rankdata(vals[i, :]) for i in range(3)])
        n_items = 8; n_raters = 3
        S = np.sum((np.sum(ranks, axis=0) - n_raters * (n_items + 1) / 2) ** 2)
        w_val = 12 * S / (n_raters ** 2 * (n_items ** 3 - n_items))
        # χ² and p (Friedman approximation: χ² = n_raters × (n_items - 1) × W)
        from scipy.stats import chi2 as chi2_dist
        chi2_val = n_raters * (n_items - 1) * w_val
        p_val = 1 - chi2_dist.cdf(chi2_val, df=n_items - 1)

        im = ax.imshow(rho_mat, cmap=WONG_DIVERGING, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(ST_MODEL_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(ST_MODEL_LABELS)
        for i in range(3):
            for j in range(3):
                v = rho_mat[i, j]
                txt_color = "white" if abs(v) > 0.55 else "black"
                fw = "bold" if (v < 0 or abs(v) > 0.9) else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=11, color=txt_color, fontweight=fw)
        if w_val >= 0.7: verdict_color = "#2E7D32"; verdict = T("Strong", "强", lang)
        elif w_val >= 0.5: verdict_color = "#F57C00"; verdict = T("Moderate", "中", lang)
        else: verdict_color = "#C62828"; verdict = T("Weak", "弱", lang)
        ax.set_title(f"({'abcd'[ci]}) {label}\nW = {w_val:.3f} ({verdict}, p = {p_val:.3g})",
                     loc="left", fontweight="bold", color=verdict_color)
    plt.tight_layout(rect=(0, 0, 0.92, 1))
    cbar_ax = fig.add_axes((0.94, 0.12, 0.018, 0.75))
    fig.colorbar(im, cax=cbar_ax, label=T("Spearman $\\rho$", "Spearman $\\rho$", lang))
    fn = f"F4_spearman_st_3x3{'_cn' if lang == 'cn' else ''}"
    fig.savefig(OUT / f"{fn}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{fn}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"F4-ST ({lang}) saved")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for lang in ["en", "cn"]:
        print(f"\n===== Rendering {lang.upper()} figures =====")
        # Standalone-chapter (§6) figures: 4-model versions
        make_f1(lang)
        make_f2(lang)
        make_f3(lang)
        make_f4(lang)
        # Main-study (§3) figures: 3 sentence-transformers only
        make_f2_st(lang)
        make_f3_st(lang)
        make_f4_st(lang)
    print("\n===== All files =====")
    import subprocess
    print(subprocess.check_output(f"ls -la {OUT}/ | grep -v '^d'", shell=True).decode())
