import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle
from matplotlib.ticker import MaxNLocator,MultipleLocator,FormatStrFormatter
# ====== 設定 ======
FILE_FOLDER = "./result/[blurkernel Proposed_PnPFBS] 2025-09-15_19-06-44/"
FILE_PATH = FILE_FOLDER + "SUMMARY(20250915 190644 754112)_both_alphan.csv"
FILE_SAVE = FILE_FOLDER + "rev2_4.eps"
ALLOWED_NOISES = [0.0025, 0.005, 0.01, 0.02, 0.04]
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
FIG_WIDTH = 14    # 横幅
FIG_HEIGHT_PER_ROW = 3.5  # 1行あたりの縦幅
# ノイズごとの y 軸範囲（なければ自動）
YLIM_DICT = {
    0.0025: (10, 40),
    0.005:  (10, 40),
    0.01:   (10, 40),
    0.02:   (10, 37),
    0.04:   (5, 32),
}

# ====== フォント設定（serifに統一）======
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 16 # 全体のフォントサイズが変更されます。
#   plt.rcParams['xtick.labelsize'] = 12 # 軸だけ変更されます。
#   plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams["legend.fancybox"] = False # 丸角
plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
#    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
#    plt.rcParams["legend.labelspacing"] = 5. # 垂直方向の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる


# ====== カーネル別のマーカー／色の指定 ======
# 好みで変更してください（未指定は自動割当）
# 好みで変更してください（未指定は自動割当）
KERNEL_MARKERS = {
    "blur_1": "o",        # 丸
    "blur_2": "^",        # 上三角
    "blur_3": "s",        # 四角
    "blur_4": "D",        # ひし形
    "blur_5": "v",        # 下三角
    "blur_6": "P",        # 6角
    "blur_7": "X",        # X
    "blur_8": "*",        # 星
    "gaussian_1_6": "h",   # ガウシアンカーネル(σ=1.6) → X印
    "square_7": ">",      # 正方形カーネル(サイズ=7) → 星印
}
KERNEL_COLORS = {
    # 好みの色名/HEXで指定（未指定はデフォルトサイクル）
    "blur_1": "#1f77b4",
    "blur_2": "#ff7f0e",
    "blur_3": "#2ca02c",
    "blur_4": "#d62728",
    "blur_5": "#9467bd",
    "blur_6": "#8c564b",
    "blur_7": "#e377c2",
    "blur_8": "#7f7f7f",
    "gaussian_1_6": "#17becf",  # ガウシアン(σ=1.6)
    "square_7": "#bcbd22",     # 正方形(サイズ=7)
}

# 予備の循環（未指定カーネルに使う）
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

    # 手法ごとにデータ抽出
    df_pnp  = df[(df["method"]=="A-PnPFBS-DnCNN") & df["blur_kernel"].isin(KERNEL_WHITELIST)]
    df_prop = df[(df["method"]=="A-Proposed")     & df["blur_kernel"].isin(KERNEL_WHITELIST)]

    # 5行×2列レイアウト
    fig, axes = plt.subplots(
        nrows=len(ALLOWED_NOISES),
        ncols=2,
        figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * len(ALLOWED_NOISES)),
        squeeze=False
    )    #fig.suptitle("Performance vs Parameter per Noise (Left: PnP-FBS / Right: Proposed)", y=0.98)

        # まとめ用の凡例ハンドル
    legend_items = {}

    def plot_panel(ax, sub, method_name, noise, xcol="alpha_n", ylim=None):
        """method_nameに応じてタイトルとx軸ラベルを切り替える"""
        sub = sub.dropna(subset=[xcol, PSNR_COL]).copy()

        # タイトルとxラベルを決定
        if method_name == "A-PnPFBS-DnCNN":
            title = rf"PnP-FBS ($\sigma={noise:g}$)"
            xlabel = r"$\zeta$"
        elif method_name == "A-Proposed":
            title = rf"Proposed ($\sigma={noise:g}$)"
            xlabel = r"$\alpha$"
        else:
            title = rf"{method_name} ($\sigma={noise:g}$)"
            xlabel = xcol

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
            # 凡例はカーネル名ごとに1つだけ保持
            legend_items.setdefault(kernel, ln)

        ax.set_title(title); ax.set_xlim(*XLIM)
        if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel); ax.set_ylabel(PSNR_COL); ax.grid(True)
        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        # ラベルを小数1桁で統一
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # 各ノイズ行を描画
    for i, noise in enumerate(ALLOWED_NOISES):
        ylim = YLIM_DICT.get(noise, None)
        plot_panel(axes[i,0], df_pnp [df_pnp ["Gaussian_noise"]==noise], "A-PnPFBS-DnCNN", noise, ylim=ylim)
        plot_panel(axes[i,1], df_prop[df_prop["Gaussian_noise"]==noise], "A-Proposed",      noise, ylim=ylim)

    # 図下に凡例を集約（色とマーカーを固定したまま表示）
    # 図下に凡例を集約（色とマーカーを固定したまま表示）
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


    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.subplots_adjust(hspace=0.5, wspace=0.25)

    out_path = FILE_SAVE
    plt.savefig(out_path, dpi=300, bbox_inches="tight")  # 高解像度 & 余白調整
    print(f"saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
