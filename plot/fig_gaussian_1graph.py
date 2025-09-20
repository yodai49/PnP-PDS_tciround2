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
    plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    plt.rcParams["figure.figsize"] = (18, 6)

#    filename = './result/result-A-20240319/DATA_A-RED-DnCNN_blur_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda0.400.npy'
#    filename = './result/result-A-20240319/DATA_A-PnPPDS-unstable-DnCNN_random_sampling_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy'
#    filename = './result/result-A-20240319/DATA_A-PnPFBS-DnCNN_random_sampling_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda1.2000000000000002.npy'
#    filename = './result/result-A-20240319/DATA_A-Proposed_random_sampling_0.010_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy'
    filename_list = [
#        './result/result-A-20240730/DATA_A-PnPFBS-DnCNN_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda1.990.npy',
#        './result/result-A-20240730/DATA_A-PnPFBS-DnCNN_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda1.990.npy',
         './result/result-test/DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_blur_0.010_300_(ILSVRC2012_val_00002289.JPEG.png)_alpha10000_lambda10000_gamma10.100_gamma24.999_max_iter1200.npy',
         './result/result-test/DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_blur_0.010_300_(ILSVRC2012_val_00002289.JPEG.png)_alpha10000_lambda10000_gamma10.100_gamma24.999_max_iter1200.npy',
#        './result/result-test/DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_blur_0.010_300_(ILSVRC2012_val_00002289.JPEG.png)_alpha10000_lambda10000_gamma10.400_gamma21.249_max_iter1200.npy',
#        './result/result-A-20240730/DATA_A-PnPPDS-DnCNN-wo-constraint_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.920_lambda10000.npy',
#        './result/result-A-20240730/DATA_A-Proposed_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.920_lambda10000.npy',
#        './result/result-A-20240730/DATA_A-PnPPDS-unstable-DnCNN_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy',
#        './result/result-A-20240730/DATA_A-PnPPDS-DnCNN-wo-constraint_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy',
#        './result/result-A-20240730/DATA_A-Proposed_random_sampling_0.005_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.8200000000000001_lambda10000.npy'
    ] 
    method_list = [
 #       'PnP-PDS (Unstable)',
#        'PnP-PDS (w/o a box const.)',
        'Proposed',
        'Proposed',
#        'Proposed',
    ]
    plotColor = ['#00D', '#D00', '#0D0', '#D0D', '#DD0', '#00D0', '#D00D']

    #filename_list = []
    #method_list = []

    #sigma_J_list = [0.0075, 0.01, 0.05, 0.1, "mixed0.1"]
    #for i in sigma_J_list:
    #    filename_list.append(
    #        f'./result/result-TCI-reply-discussion1/DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_{i}_dict_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha10000_lambda10000.npy'
    #    )
    #    method_list.append(f'$\sigma_J={i}$')
    data_list = []
    for filename in filename_list:
        data_list.append(np.load(filename, allow_pickle=True).item())

    # plot
#    fig = plt.figure()
    fig, axes = plt.subplots(1, len(filename_list), tight_layout=True)


#    fig.set_xlabel("$\\varepsilon$")
#       fig.set_ylabel("PSNR [dB]")

    for index, each_method in enumerate(data_list):
        for index2, each_data in enumerate(each_method['results'].values()):
            if (index2 == 0):
                label = method_list[index]
            else:
                label = ''
            y = each_data['PSNR_evolution']
            axes[0].plot(y,  color=plotColor[index], label = label)
            y = each_data['c_evolution']
            axes[1].plot(y,  color=plotColor[index])

#        axes[0,index].set_title(method_list[index])
        axes[index].grid(color="gainsboro")
        axes[index].set_xlabel("iteration $n$")
    axes[0].set_ylim(8, 40)
    axes[0].set_ylabel("PSNR")
    #axes[1].set_ylim(pow(10,-5)*0.8, pow(10, -0.5))
    axes[1].set_yscale('log')
    axes[1].set_ylabel("$c_n$")

    # save
    fig.legend(loc='lower center', ncol=2, bbox_to_anchor = (0.5, 0), handletextpad = 0.7)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    fig.savefig('./result/result-test/graph_gaussian_evolution_blur_2.png', bbox_inches="tight", pad_inches=0.1)
    fig.savefig('./result/result-test/graph_gaussian_evolution_blur_2.eps', bbox_inches="tight", pad_inches=0.1)




if (__name__ == '__main__'):
    plot_graph_evolution()