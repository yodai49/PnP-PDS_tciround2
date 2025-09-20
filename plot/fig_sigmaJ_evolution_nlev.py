import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_graph_evolution():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
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
    plt.rcParams["legend.handletextpad"] = 1. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    plt.rcParams["figure.figsize"] = (22, 6)

    # パラメータ設定
    nlev_list = ['0.0075', '0.01', '0.05', '0.1', 'mixed0.1']
    nlev_list_legened = [
        '$\\sigma_J=0.0075$', '$\\sigma_J=0.01$',
        '$\\sigma_J=0.05$', '$\\sigma_J=0.1$',
        '$\\sigma_J\\in[0,\,1]$'
    ]
    plotColor = ['#4784BF', '#39A869', '#F2E55C', '#DE6641', '#999']
    noise_levels = ['0.0025', '0.010', '0.100', '0.250']
    task = 'blur'

    # 2x2のサブプロット作成
    fig, axes = plt.subplots(2, 2, figsize=(22, 8), tight_layout=True)

    for idx, noise_level in enumerate(noise_levels):
        row, col = divmod(idx, 2)  # 左上→右下の順に配置
        ax = axes[row, col]

        for nlev_idx, nlev in enumerate(nlev_list):
            method_label = nlev_list_legened[nlev_idx]
            color = plotColor[nlev_idx]

            filename = (
                f'./result/result-TCI-reply-discussion1/Gaussian/'
                f'DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_{nlev}_dict_{task}_{noise_level}_300_'
                f'(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda10000.npy'
            )

            try:
                method_data = np.load(filename, allow_pickle=True).item()
            except FileNotFoundError:
                print(f"File not found: {filename}")
                continue

            psnr_all = [d['PSNR_evolution'] for d in method_data['results'].values()]
            psnr_avg = np.mean(psnr_all, axis=0)

            label = method_label if idx == 0 else None  # 凡例は左上だけに出す
            ax.plot(psnr_avg, color=color, label=label, linewidth=2.5)

        # 軸設定
        ax.set_title(f"$\sigma={noise_level}$")
        ax.set_xlabel("iteration $n$")
        ax.set_ylabel("PSNR")
        ax.set_ylim(12, 38)
        ax.grid(color="gainsboro")

    # 上2つのグラフの x 軸ラベルと目盛りを非表示
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(labelbottom=False)

    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(labelbottom=False)
    # 凡例を図の下にまとめる
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(nlev_list), fontsize=10, title="$\\sigma_J$")

    # save
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(legend_labels),  # 横並び
        title=None,
    )
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    fig.savefig('./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_evolution.png', bbox_inches="tight", pad_inches=0)
    fig.savefig('./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_evolution.eps', bbox_inches="tight", pad_inches=0)


if (__name__ == '__main__'):
    plot_graph_evolution()