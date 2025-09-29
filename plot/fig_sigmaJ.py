import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def apply_bar_edge_and_fill(ax, edge_colors):
    # ax.containers は系列ごとの BarContainer（描画順で並ぶ）
    for i, container in enumerate(ax.containers):
        col = edge_colors[i % len(edge_colors)]
        r, g, b, _ = to_rgba(col)
        for patch in container.patches:
            patch.set_edgecolor((r, g, b, 1.0))  # 外枠＝不透明
            patch.set_linewidth(1.3)
            patch.set_facecolor((r, g, b, 0.8))  # 中＝同色でα=0.2
# LaTeXラベルへの変換用マッピング
replacement_map = {
    "reply_DnCNN_nobn_nch_3_nlev_0.0075_dict": r"$\sigma_J=0.0075$",
    "reply_DnCNN_nobn_nch_3_nlev_0.01_dict": r"$\sigma_J=0.01$",
    "reply_DnCNN_nobn_nch_3_nlev_0.05_dict": r"$\sigma_J=0.05$",
    "reply_DnCNN_nobn_nch_3_nlev_0.1_dict": r"$\sigma_J=0.1$",
    "reply_DnCNN_nobn_nch_3_nlev_mixed0.1_dict": r"$\sigma_J\in[0,\,0.1]$"
}

# 新しい "sigmaJ" 列を作成
def extract_sigmaJ_label(name: str) -> str:
    for key, val in replacement_map.items():
        if key in name:
            return val
    return "unknown"

# "sigmaJ" 列を追加
def prepare_pivot(df, content):
    # ピボットテーブルを作成（最大値）
    pivot_df = df.pivot_table(
        index='Gaussian_noise',
        columns='sigmaJ',
        values=content,
        aggfunc='max'
    )

    # 整形と型変換
    pivot_df = pivot_df.reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.astype({col: "float" for col in pivot_df.columns if col != "Gaussian_noise"})

    return pivot_df

def plot_graph():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 16 # 全体のフォントサイズが変更されます。
 #   plt.rcParams['xtick.labelsize'] = 12 # 軸だけ変更されます。
 #   plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.2 # axis line width
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = '#CCC' # edgeの色を変更
#    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
#    plt.rcParams["legend.labelspacing"] = 5. # 垂直方向の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    plt.rcParams["figure.figsize"] =  (18, 7)

    file_path = "./result/result-TCI-reply-discussion1/Gaussian/SUMMARY(20250604 100701 665848).txt"

    # 1行目から列名だけ読み取る
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')

    desired_cols = ["Observation", "Gaussian_noise", "denoiser(architecture)", "PSNR", "SSIM"]
    col_indices = [i for i, col in enumerate(header) if col.strip() in desired_cols]
    df = pd.read_csv(file_path, usecols=col_indices, skiprows=1, header=None, names=[header[i] for i in col_indices])
    df.columns = df.columns.str.strip()
    df["Observation"] = df["Observation"].astype(str).str.strip()
    df = df.iloc[:-1]
    df["sigmaJ"] = df["denoiser(architecture)"].apply(extract_sigmaJ_label)
    df = df[~df["Gaussian_noise"].str.contains("0.0025")]
    df = df[~df["Gaussian_noise"].str.contains("0.25")]
    # 分割
    df_blur = df[df["Observation"] == "blur"]
    df_rs = df[df["Observation"] == "random_sampling"]
     # ピボットテーブル形式に集計（最大値を取る）
    pivot_blur_psnr = prepare_pivot(df_blur, "PSNR")
    pivot_rs_psnr = prepare_pivot(df_rs, "PSNR")
    pivot_blur_ssim = prepare_pivot(df_blur, "SSIM")
    pivot_rs_ssim = prepare_pivot(df_rs, "SSIM")

    fig, axes = plt.subplots(2, 2, figsize=(20, 7))

    # blurプロット
    barWidth = 0.95
    color_list = ["#65418B", "#518BD2", "#48772D", "#EED948", "#FC3D34", "#B00000", "#A00000"]

    colors = ["#4156F7DD", "#6FCC39DD", "#F3DB40DD","#ED5050DD", '#777D']  # 色を指定

    pivot_blur_psnr.plot.bar(
        x='Gaussian_noise',
        y=pivot_blur_psnr.columns[1:],
        ax=axes[0][0],
        title="Deblurring",
        xlabel=None,
        ylabel="PSNR",
        width = barWidth,
        color=colors
    )

    # rsプロット
    pivot_rs_psnr.plot.bar(
        x='Gaussian_noise',
        y=pivot_rs_psnr.columns[1:],
        ax=axes[0][1],
        title="Inpainting",
        xlabel=None,
        ylabel="PSNR",
        width = barWidth,
        color=colors
    )
        
    pivot_blur_ssim.plot.bar(
        x='Gaussian_noise',
        y=pivot_blur_ssim.columns[1:],
        ax=axes[1][0],
        title=None,
        xlabel="$\sigma$",
        ylabel="SSIM",
        width = barWidth,
        color=colors
    )

    # rsプロット
    pivot_rs_ssim.plot.bar(
        x='Gaussian_noise',
        y=pivot_rs_ssim.columns[1:],
        ax=axes[1][1],
        title=None,
        xlabel="$\sigma$",
        ylabel="SSIM",
        width = barWidth,
        color=colors
    )
    for ax in axes.flat:
        ax.legend_.remove()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, axis='y',  linewidth=0.5, color='gray', alpha=0.4)
        ax.grid(True, axis='x',  linewidth=0.5, color='gray', alpha=0.4)
        ax.set_axisbelow(True)
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()

    plt.subplots_adjust(bottom=0.25)

    axes[0, 0].set_ylim(8, 40)
    axes[0, 1].set_ylim(8, 40)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    lefts = []
    rights = []
    for ax in axes.flat:
        bbox = ax.get_position()
        lefts.append(bbox.x0)
        rights.append(bbox.x1)
    left = min(lefts)
    right = max(rights)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(left, 0.09, right - left, 0.05),  # (x, y, width, height)
        mode="expand",
        ncol=len(legend_labels),
    )
    plt.show()

    fig.savefig('./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_performance.png', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_performance.eps', bbox_inches="tight", pad_inches=0.05)


if (__name__ == '__main__'):
    plot_graph()