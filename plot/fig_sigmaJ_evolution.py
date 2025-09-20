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

    nlev_list = ['0.0075', '0.01', '0.05', '0.1', 'mixed0.1']
    nlev_list_legened = [
        '$\sigma_J=0.0075$', '$\sigma_J=0.01$',
        '$\sigma_J=0.05$', '$\sigma_J=0.1$',
        '$\sigma_J\in[0,\,1]$'
    ]
    plotColor = ['#4784BF', '#39A869', '#F2E55C','#DE6641', '#999']

    task_list = ['blur', 'random_sampling']  # ← blurとinpainting両方

    fig, axes = plt.subplots(2, 2, figsize=(17, 8), tight_layout=True)

    for row, task in enumerate(task_list):
        for index, nlev in enumerate(nlev_list):
            method_label = nlev_list_legened[index]
            color = plotColor[index]

            filename = (
                f'./result/result-TCI-reply-discussion1/Gaussian/'
                f'DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_{nlev}_dict_{task}_0.0025_300_'
                f'(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda10000.npy'
            )

            # データ読み込み
            try:
                method_data = np.load(filename, allow_pickle=True).item()
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {filename}")
                continue

            # 平均を計算
            psnr_all = []
            c_all = []
            for each_data in method_data['results'].values():
                psnr_all.append(each_data['PSNR_evolution'])
                c_all.append(each_data['c_evolution'])

            psnr_avg = np.mean(psnr_all, axis=0)
            c_avg = np.mean(c_all, axis=0)

            # 描画
            label = method_label
            axes[row, 0].plot(psnr_avg, color=color, label=label)
            axes[row, 1].plot(c_avg, color=color)

    # 軸ラベルと凡例・グリッド
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()

    titles = ['PSNR evolution', '$c$ evolution']
    for i in range(2):
        for j in range(2):
            axes[i, j].grid(color="gainsboro")
            axes[i, j].set_xlabel("iteration $n$")
#            axes[i, j].legend_.remove()
        axes[i, 0].set_ylabel("PSNR")
        axes[i, 0].set_ylim(8, 40)
        axes[i, 1].set_ylabel("$c_n$")
        axes[i, 1].set_yscale('log')


    #axes[0].set_ylim(8, 40)
    #axes[0].set_ylabel("PSNR")
    axes[0, 1].set_yscale('log')
    #axes[1].set_ylabel("$c_n$")

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
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    fig.savefig('./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_evolution.png', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_evolution.eps', bbox_inches="tight", pad_inches=0.05)


if (__name__ == '__main__'):
    plot_graph_evolution()