import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle
from math import ceil
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

# ====== 設定 ======
FILE_FOLDER = "./result/[blurkernel Proposed_PnPFBS] 2025-09-15_19-06-44/"
FILE_PATH = FILE_FOLDER + "SUMMARY(20250915 190644 754112)_both_alphan.csv"
FILE_SAVE = FILE_FOLDER + "rev2_4_proposed_only.eps"

# 表示したいノイズレベル
ALLOWED_NOISES = [0.0025, 0.005, 0.01, 0.02, 0.04]

# レイアウト（2列×3行に固定しつつ、必要分だけ使用）
N_COLS = 2
N_ROWS = 3  # 2×3=6 パネル（5つ使って残り1つは非表示）

KERNEL_WHITELIST = ["blur_1","blur_2","blur_3","blur_4","blur_5","blur_6","blur_7","blur_8", "gaussian_1_6", "square_7"]
KERNEL_LABELS = {
    "blur_1": "(a)",
    "blur_2": "(b)",
    "blur_3": "(c)",
    "blur_4": "(d)",
    "blur_5": "(e)",
    "blur_6": "(f)",
    "blur_7": "(g)",
    "blur_8": "(h)",
    "gaussian_1_6": "(i)",
    "square_7": "(j)",
}

PSNR_COL = "PSNR"
XLIM = (0.75, 1.3)

FIG_WIDTH = 12
FIG_HEIGHT = 12  # 2×3で見やすい全体高さ

# ノイズごとの y 軸範囲（なければ自動）
YLIM_DICT = {
    0.0025: (10, 40),
    0.005:  (10, 40),
    0.01:   (10, 40),
    0.02:   (10, 37),
    0.04:   (5, 32),
}

# ====== フォント設定（serifに統一）======
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 16
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.handletextpad"] = 3.
plt.rcParams["legend.markerscale"] = 1
plt.rcParams["legend.borderaxespad"] = 0.

# ====== カーネル別のマーカー／色の指定 ======
KERNEL_MARKERS = {
    "blur_1": "o", "blur_2": "^", "blur_3": "s", "blur_4": "D",
    "blur_5": "v", "blur_6": "P", "blur_7": "X", "blur_8": "*",
    "gaussian_1_6": "h", "square_7": ">"
}
KERNEL_COLORS = {
    "blur_1": "#1f77b4",  "blur_2": "#ff7f0e", "blur_3": "#2ca02c",
    "blur_4": "#d62728",  "blur_5": "#9467bd","blur_6": "#8c564b",
    "blur_7": "#e377c2",  "blur_8": "#7f7f7f","gaussian_1_6": "#17becf",
    "square_7": "#bcbd22",
}
FALLBACK_MARKERS = cycle(["o", "^", "s", "D", "v", "P", "X", "*", "<", ">", "h", "H"])
FALLBACK_COLORS  = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))

def get_style_for_kernel(kernel: str):
    marker = KERNEL_MARKERS.get(kernel, next(FALLBACK_MARKERS))
    color  = KERNEL_COLORS.get(kernel,  next(FALLBACK_COLORS))
    return marker, color

def main():
    df = pd.read_csv(FILE_PATH, engine="python")

    # 必須列のみ残して数値化
    need = ["Gaussian_noise","blur_kernel","method","myLambda","alpha_n",PSNR_COL]
    df = df[need].copy()
    for c in ["Gaussian_noise","myLambda","alpha_n",PSNR_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["blur_kernel"] = df["blur_kernel"].astype(str).str.strip()

    # Proposed のみ抽出
    df_prop = df[(df["method"]=="A-Proposed") & df["blur_kernel"].isin(KERNEL_WHITELIST)]

    # 図とAxes（2列×3行）
    fig, axes = plt.subplots(
        nrows=N_ROWS, ncols=N_COLS,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        squeeze=False
    )

    # まとめ用の凡例ハンドル
    legend_items = {}

    def plot_panel(ax, sub, noise, ylim=None, xcol="alpha_n"):
        """Proposedの1パネルを描く"""
        sub = sub.dropna(subset=[xcol, PSNR_COL]).copy()

        title = rf"Proposed ($\sigma={noise:g}$)"
        xlabel = r"$\alpha$"

        if sub.empty:
            ax.set_title(title); ax.set_xlim(*XLIM)
            if ylim: ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel); ax.set_ylabel(PSNR_COL); ax.grid(True)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        for kernel, g in sub.groupby("blur_kernel"):
            g = g.sort_values(xcol)
            marker, color = get_style_for_kernel(kernel)
            ln, = ax.plot(
                g[xcol], g[PSNR_COL],
                marker=marker, linewidth=1.2, markersize=4,
                label=kernel, color=color,
            )
            legend_items.setdefault(kernel, ln)

        ax.set_title(title); ax.set_xlim(*XLIM)
        if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel); ax.set_ylabel(PSNR_COL); ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # 各ノイズを 2×3 に敷き詰めて描画（余りは非表示）
    total_slots = N_ROWS * N_COLS
    for idx in range(total_slots):
        r, c = divmod(idx, N_COLS)
        ax = axes[r, c]
        if idx < len(ALLOWED_NOISES):
            noise = ALLOWED_NOISES[idx]
            sub = df_prop[df_prop["Gaussian_noise"] == noise]
            ylim = YLIM_DICT.get(noise, None)
            plot_panel(ax, sub, noise, ylim=ylim)
        else:
            ax.set_visible(False)

    # 図下に凡例を集約
    if legend_items:
        fig.legend(
            list(legend_items.values()),
            [KERNEL_LABELS.get(k, k) for k in legend_items.keys()],
            loc="lower center",
            ncol=len(legend_items),
            frameon=True, framealpha=1.0, edgecolor="black",
            borderpad=0.7, labelspacing=0.8, handletextpad=0.6, columnspacing=1.1,
            fontsize=15, bbox_to_anchor=(0.5, 0.02)
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    out_path = FILE_SAVE
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
