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
    plt.rcParams["figure.figsize"] =  (18, 6)

    filename_list = [
#        './result/result-C-20240812/DATA_C-PnPADMM-DnCNN_blur_00000_100_(03.png)_alpha10000_lambda0.200.npy',
#        './result/result-C-20240812/DATA_C-PnPPDS-DnCNN-wo-constraint_blur_00000_100_(03.png)_alpha10000_lambda0.001.npy',
#        './result/result-TCI-reply-discussion2/Poisson/DATA_C-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.0075_dict_blur_00000_100_(01.png)_alpha10000_lambda0.00125_gamma10.03000.npy',
#        './result/result-TCI-reply-discussion2/Poisson/DATA_C-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.0075_dict_blur_00000_100_(01.png)_alpha10000_lambda0.00125_gamma10.03000.npy'
#        './result/result-C-20240812/DATA_C-Proposed_blur_00000_100_(03.png)_alpha10000_lambda0.00125.npy',
#        './result/result-C-20240406/DATA_C-PnPADMM-DnCNN_random_sampling_00000_100_(03.png)_alpha10000_lambda0.400.npy',
        './result/2025-09-20_21-33-16/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
        './result/2025-09-20_21-33-16/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
    ] 
    data_list = []
    for filename in filename_list:
        data_list.append(np.load(filename, allow_pickle=True).item())
    method_list = [
#        'PnP-ADMM',
#        'PnP-PDS w/o box constraint',
        'PnP-PDS (Unstable)',
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
        axes[0,index].set_ylim(18, 29)
        axes[0,index].set_ylabel("PSNR")
        axes[1,index].set_ylim(pow(10,-7), pow(10, -1))
        axes[1,index].set_yscale('log')
        axes[1,index].set_ylabel("$c_n$")
        for i in range(0, 2):
            axes[i,index].grid(color="gainsboro")
            axes[i,index].set_xlabel("iteration $n$")

    # save
    plt.show()
    #fig.savefig('./result/result-C-20241119(proposed-revise)/graph_poisson_evolution.png', bbox_inches="tight", pad_inches=0.05)
    #fig.savefig('./result/result-C-20241119(proposed-revise)/graph_poisson_evolution.eps', bbox_inches="tight", pad_inches=0.05)




if (__name__ == '__main__'):
    plot_graph_evolution()