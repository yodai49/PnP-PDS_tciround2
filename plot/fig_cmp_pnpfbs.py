import matplotlib.pyplot as plt
import numpy as np
import math
import csv

def get_denoiser_nl(denoiser):
    if (denoiser == 'DnCNN(reply_DnCNN_nobn_nch_3_nlev_0.005_dict)'):
        nl_den = 0.005
    elif (denoiser == 'DnCNN(DnCNN_nobn_nch_3_nlev_0.01_dict)'):
        nl_den = 0.01
    else:
        return -1 # Invalid architecture
    return nl_den

def get_optimal_lambda(denoiser, gaussian_nl, h_norm):
    nl_den = get_denoiser_nl(denoiser)
    return 1 / (2 * h_norm) * (nl_den / gaussian_nl)

def get_sigma_coef(denoiser, sigma, h_norm):
    nl_den = get_denoiser_nl(denoiser)
    return round(sigma * (4 * h_norm) / 0.01, 3)

def plot_graph():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 16 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
#    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
#    plt.rcParams["legend.labelspacing"] = 5. # 垂直方向の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 0.5 # 凡例の線と文字の距離の長さ
    #plt.rcParams["legend.markerscale"] = 0 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    plt.rcParams["figure.figsize"] = (18, 4.5) # グラフのサイズを指定

    filename = './result/result-TCI-reply-A1/SUMMARY(20250528 133224 572588).txt'
    with open(filename) as f:
        reader = csv.reader(f)
        l = [row for row in reader]

    l = l[1:]
    l = l [:-1]

    data = {}
    h_norm = 0.225

    for line in l:
        phi = line[0]
        gaussian_nl = line[1]
        method = line[3]
        architecture = line[5]
        psnr_average = line[6]
        ssim_average = line[7]
        if (method == 'A-Proposed'):
            x = line[11]
        elif (method == 'A-PnPFBS-DnCNN'):
            x = line[12]

        if (method == 'A-Proposed' or method == 'A-PnPFBS-DnCNN'):
            if (not (method in data)):
                data[method] = {}
            if (not (architecture in data[method])):
                data[method][architecture] = {}
            if (not (gaussian_nl in data[method][architecture])):
                data[method][architecture][gaussian_nl] = {'x' : [], 'psnr' : []}
            data[method][architecture][gaussian_nl]['psnr'].append(float(psnr_average))
        if (method == 'A-Proposed'):
            data[method][architecture][gaussian_nl]['x'].append(float(x))
        elif (method == 'A-PnPFBS-DnCNN'):
            data[method][architecture][gaussian_nl]['x'].append(float(x) / get_optimal_lambda(architecture, float(gaussian_nl), h_norm))

    denoiser_list = ['DnCNN(DnCNN_nobn_nch_3_nlev_0.01_dict)']
    denoiser_list_disp = ['0.01']
    method_list = ['A-PnPFBS-DnCNN', 'A-Proposed']
    method_list_disp = ['PnP-FBS', 'Proposed']

#    fig = plt.figure()
    fig, axes = plt.subplots(1,len(method_list))

#    fig.set_xlabel("$\\varepsilon$")
#       fig.set_ylabel("PSNR [dB]")

    dic_box = {
        'facecolor' : 'white',
        'edgecolor' : 'black',
        'linewidth' : 1
    }
    # 固定カラーとマーカーの配列を用意  
    color_list = ["#65418B", "#518BD2", "#415734", "#EED948", "#FC3D34", "#B00000", "#A00000"]
    marker_list = ['o', 's', '^', 'v', 'D', 'P', '*']  # 丸, 四角, 三角, 下三角, 菱形, 六角, 星など

    for index0, each_method in enumerate(method_list):
        each_axis = axes[index0]
        each_axis.grid(True, axis='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

        each_axis.set_title(method_list_disp[index0] + " (Deblurring)")
        if each_method == 'A-Proposed':
            each_axis.set_xlabel("$\\alpha$")
        elif each_method == 'A-PnPFBS-DnCNN':
            each_axis.set_xlabel("$\\zeta$")
        
        each_axis.set_ylabel("PSNR")
        each_axis.set_ylim(17, 39)
        each_axis.set_xlim(0.7, 1.25)

        for plot_index, (each_key, each_plot_data) in enumerate(data[each_method][denoiser_list[0]].items()):
            x = each_plot_data['x']
            y = each_plot_data['psnr']
            sigma = float(each_key)

            # 安全なインデックス取得（配列より長い場合はmodでループ）
            color = color_list[plot_index % len(color_list)]
            marker = marker_list[plot_index % len(marker_list)]

            each_axis.plot(
                x, y,
                color=color,
                marker=marker,
                markersize=6,
                markevery=1,
                markeredgewidth=1.0,
                markeredgecolor='k',
                label=f"$\\sigma={sigma}$",
                linewidth=2.0,
                alpha=0.8
            )
    #cmap = plt.get_cmap("rainbow") # カラーマップ"Blues"を取得
    plt.subplots_adjust(bottom=0.3)
    fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),  # 位置調整: 少し上にする（-0.05 → -0.15 などで様子を見る）
        ncol=len(data[method_list[0]][denoiser_list[0]]),
        frameon=True,
        fontsize=14
    )
    # save
    plt.show()
    fig.savefig('./result/result-TCI-reply-A1/Cmp_PnPFBS.png', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('./result/result-TCI-reply-A1/Cmp_PnPFBS.eps', bbox_inches="tight", pad_inches=0.05)


if (__name__ == '__main__'):
    plot_graph()