import matplotlib.pyplot as plt
import numpy as np
import cv2

def view_npy(path):
    data = np.load(path, allow_pickle=True).item()
    gamma1 = data['method']['gamma1']
    data = data["results"]
    #fne_data = data['other_data']['fne_data']
#    print(picture_all)
#    print(data[0]['other_data'])
    print(1 * np.linalg.norm(data[0]['other_data']['y1'].flatten(), ord=2))
    print(np.linalg.norm(data[0]['other_data']['n'].flatten(), ord=2))
    y1 = data[0]['other_data']['y2']
    n = -data[0]['other_data']['n']
    plt.figure(figsize=(6, 4))
    plt.scatter(n, y1, color='blue', edgecolor='k', alpha=0.7)
    plt.xlabel("n")
    plt.ylabel("y1")
    plt.title("Scatter Plot of y1 vs. n")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    every = 1
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    fig_1.set_xlabel(r"iterations")
    fig_1.set_ylabel('Negative_ratio')
    for i in range(0, 1):
        fne_each = data[i]['other_data']['fne_data'][::every]
        x = np.linspace(0, fne_each.size, (int)(fne_each.size))
        fig_1.plot(x, fne_each, marker='o', markersize=0, markevery = 1, markeredgewidth=1., markeredgecolor='k', label=str(i))
#        fig_1.set_ylim(0, 100)
        fig_1.set_yscale('log')

    plt.grid(color="gainsboro")

#    fig_1.legend(ncol=1, bbox_to_anchor=(0., 1.025, 1., 0.102), loc="lower right")
    fig_1.legend(ncol=1, loc="best")

    # save
    plt.show()

    return
    picture_all = picture_all.reshape([3001, 3, 128, 128])
    for i in range(0, 300):
        picture_temp = picture_all[i * 10 + 1]    
        if(np.ndim(picture_temp) == 3):
            # color
            picture_temp = np.moveaxis(picture_temp, 0, 2)
        picture_temp[picture_temp > 1.] = 1.
        picture_temp[picture_temp < 0.] = 0.
        picture_temp = np.nan_to_num(picture_temp, nan=1)

        cv2.imwrite("./result/result-test/test" + str(i) + ".png", np.uint8(picture_temp*255.))

def plot_graph():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 14 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 12 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
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
    
    plot_content = 'PSNR'
    every = 1
    plot_content_data = {'PSNR' : {'title' : 'PSNR', 'key' : 'PSNR_evolution'}, 
                        'c' :  {'title' : '$c_n$', 'key' : 'c_evolution'}}
 
    filename_list = ['./result/DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_blur_0.010_300_(ILSVRC2012_val_00002289.JPEG.png)_alpha10000_lambda10000_gamma10.500_gamma20.999_max_iter1200.npy']
    y_list = [None] * len(filename_list)
    data_list = [None] * len(filename_list)
    method_name_list = [None] * len(filename_list)
    for filename in filename_list:
        data_list[filename_list.index(filename)] = np.load(filename, allow_pickle=True).item()
        y_list[filename_list.index(filename)] = data_list[filename_list.index(filename)]['results'][0][plot_content_data[plot_content]['key']]
        method_name_list[filename_list.index(filename)] = str(data_list[filename_list.index(filename)]['summary']['algorithm']) + ' (' + str(data_list[filename_list.index(filename)]['summary']['denoiser']) + ')'
    x = np.linspace(0, y_list[0].size, (int)(y_list[0].size / every))
    for filename in filename_list:
        y_list[filename_list.index(filename)] = y_list[filename_list.index(filename)][::every]

    # plot
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    fig_1.set_xlabel(r"iterations")
    fig_1.set_ylabel(plot_content_data[plot_content]['title'])
    for y in y_list:
        y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0)  # NaN/Inf 対策
        fig_1.plot(x, y,
                marker='o',
                markersize=7,
                markevery=1,
                markeredgewidth=1.,
                markeredgecolor='k',
                color="r",
                label=method_name_list[y_list.index(y)])


    fig_1.set_ylim(0, 100)
    if (plot_content == 'c'):
        plt.yscale('log')
    plt.grid(color="gainsboro")

#    fig_1.legend(ncol=1, bbox_to_anchor=(0., 1.025, 1., 0.102), loc="lower right")
    fig_1.legend(ncol=1, loc="best")

    # save
    plt.show()
#    fig.savefig('./ICASSP-result/test.png', bbox_inches="tight", pad_inches=0.05)
#    fig.savefig('./ICASSP-result/test.eps', bbox_inches="tight", pad_inches=0.05)

if (__name__ == '__main__'):
#    plot_graph()
    view_npy('./result/result-test/DATA_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_blur_0.010_300_(ILSVRC2012_val_00002289.JPEG.png)_alpha10000_lambda10000_gamma10.500_gamma20.999_max_iter1200.npy')