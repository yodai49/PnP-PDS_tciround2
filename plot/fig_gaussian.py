import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_graph_evolution():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
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
    plt.rcParams["figure.figsize"] = (18, 8)

#    filename = './result/result-A-20240319/DATA_A-RED-DnCNN_blur_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda0.400.npy'
#    filename = './result/result-A-20240319/DATA_A-PnPPDS-unstable-DnCNN_random_sampling_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy'
#    filename = './result/result-A-20240319/DATA_A-PnPFBS-DnCNN_random_sampling_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda1.2000000000000002.npy'
#    filename = './result/result-A-20240319/DATA_A-Proposed_random_sampling_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy'
    filename_list = [
#        './result/result-A-20240730/DATA_A-PnPFBS-DnCNN_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda1.990.npy',
#        './result/result-A-20240730/DATA_A-PnPFBS-DnCNN_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda1.990.npy',
        './result/result-TCI-blurkernel/DATA_A-PnPPDS-DnCNN-clipping-layer_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_square_7_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha100_lambda1000.npy',
        './result/result-TCI-blurkernel/DATA_A-PnPPDS-DnCNN-clipping-layer_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_square_7_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha100_lambda1000.npy',
        './result/result-TCI-blurkernel/DATA_A-PnPPDS-DnCNN-clipping-layer_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_square_7_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha100_lambda1000.npy',
#        './result/result-A-20240730/DATA_A-PnPPDS-unstable-DnCNN_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy',
#        './result/result-A-20240730/DATA_A-PnPPDS-DnCNN-wo-constraint_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy',
#        './result/result-A-20240730/DATA_A-Proposed_random_sampling_0.005_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy'
    ] 
    data_list = []
    for filename in filename_list:
        data_list.append(np.load(filename, allow_pickle=True).item())
    method_list = [
        'PnP-PDS (Unstable)',
        'PnP-PDS (w/o a box const.)',
        'Proposed',
    ]

    # plot
#    fig = plt.figure()
    fig, axes = plt.subplots(2, len(filename_list), tight_layout=True)


#    fig.set_xlabel("$\\varepsilon$")
#       fig.set_ylabel("PSNR [dB]")
    plotColor = '#D00'
    for index, each_method in enumerate(data_list):
        for each_data in each_method['results'].values():
            y = each_data['PSNR_evolution']
            axes[0,index].plot(y,  label='', color=plotColor)
            y = each_data['c_evolution']
            axes[1,index].plot(y,  label='', color=plotColor)

#        axes[0,index].set_title(method_list[index])
        axes[0,index].set_title(method_list[index])
        axes[0,index].set_ylim(8, 40)
        axes[0,index].set_ylabel("PSNR")
        axes[1,index].set_ylim(pow(10,-6)*0.8, pow(10, 0))
        axes[1,index].set_yscale('log')
        axes[1,index].set_ylabel("$c_n$")
        for i in range(0, 2):
            axes[i,index].grid(color="gainsboro")
            axes[i,index].set_xlabel("iteration $n$")

    # save
    plt.show()
#    fig.savefig('./result/result-A-20240730/graph_gaussian_evolution_blur_2.png', bbox_inches="tight", pad_inches=0.05)
#    fig.savefig('./result/result-A-20240730/graph_gaussian_evolution_blur_2.eps', bbox_inches="tight", pad_inches=0.05)




if (__name__ == '__main__'):
    plot_graph_evolution()