import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_graph_evolution():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 18 # 全体のフォントサイズが変更されます。
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
    plt.rcParams["legend.handletextpad"] = 2. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    plt.rcParams["figure.figsize"] = (9, 5)

    filename_list_deb001 = [ # Inpainting at sigma=0.01
        './result/result-TCI-reply-instability-all/DATA_A-PnPPDS-DnCNN-wo-constraint_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-RED-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-PnPFBS-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-Proposed_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
    ]
    filename_list_deb004 = [ # deblurring at sigma=0.04
#        'result/result-TCI-reply-instability-all/DATA_A-PnPPDS-DnCNN-wo-constraint_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
#        'result/result-TCI-reply-instability-all/DATA_A-RED-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        'result/result-TCI-reply-instability-all/DATA_A-PnPFBS-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        'result/result-TCI-reply-instability-all/DATA_A-Proposed_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',    
    ]
    filename_list_inp001 = [ # Inpainting at sigma=0.01
        './result/result-TCI-reply-instability-all/DATA_A-PnPPDS-DnCNN-wo-constraint_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-RED-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-PnPFBS-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-Proposed_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
    ]
    filename_list_deb004_fbs_lamb15_1img = [
        'result/result-TCI-reply-instability-deblurring004-fbs-lamb1.5/DATA_A-PnPPDS-DnCNN-wo-constraint_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00038157.JPEG.png)_alpha10000_lambda10000.npy',
    #    'result/result-TCI-reply-instability-deblurring004-fbs-lamb1.5/DATA_A-RED-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00038157.JPEG.png)_alpha10000_lambda10000.npy',
        'result/result-TCI-reply-instability-deblurring004-fbs-lamb1.5/DATA_A-PnPFBS-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00038157.JPEG.png)_alpha10000_lambda1.500.npy',
        'result/result-TCI-reply-instability-deblurring004-fbs-lamb1.5/DATA_A-Proposed_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.040_300_(ILSVRC2012_val_00038157.JPEG.png)_alpha10000_lambda10000.npy',
    ]
    filename_list = filename_list_inp001
    method_list = [
        'PnP-PDS (w/o a box const.)',
        'RED',
        'PnP-FBS',
        'Proposed',
    ]
    
    fne_data_averaged_list = {}
    for index, filename in enumerate(filename_list):
        each_file = np.load(filename, allow_pickle=True).item()
        data_list = []
        for index2, each_data in enumerate(each_file['results'].items()):
            each_fne_data = each_file['results'][index2]["other_data"]['fne_data']
            data_list.append(each_fne_data)
        fne_data_averaged_list[method_list[index]] = np.mean(data_list, axis=0) 

    # plot
#    fig = plt.figure()
    fig, axes = plt.subplots(1, 1, tight_layout=True)


#    fig.set_xlabel("$\\varepsilon$")
#       fig.set_ylabel("PSNR [dB]")
    plotColor = [
#        "#0000DD", 
        "#E6AB02", # PnP-PDS (w/o a box const.)
        "#1B9E77", # RED
        "#940", # PnP-FBS
        "#D00", # Proposed
    ]
    axes.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label=None, alpha=0.6)
    for index, each_data in enumerate(fne_data_averaged_list):
        y = fne_data_averaged_list[method_list[index]]
        label = method_list[index]
        axes.plot(y, color = plotColor[index] , label = label, linewidth=1.0)

#        axes[0,index].set_title(method_list[index])
        axes.grid(color="gainsboro")
        axes.set_xlabel("iteration $n$")
    axes.set_ylim(1e-1, 1e4)
    axes.set_yscale('log')
    axes.set_ylabel("$\\|\\nabla Q(\mathbf{x}_n)\\|_{\mathrm{sp}}^2$") # 2-norm

    # save
    fig.legend(loc='lower center', ncol=2, bbox_to_anchor = (0.5, 0), handletextpad = 0.7)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()

    fig.savefig('./result/result-TCI-reply-instability-all/Jacob_Q_evol_inp001_averaged.png', bbox_inches="tight", pad_inches=0)
    fig.savefig('./result/result-TCI-reply-instability-all/Jacob_Q_evol_inp001_averaged.eps', bbox_inches="tight", pad_inches=0)


if (__name__ == '__main__'):
    plot_graph_evolution()