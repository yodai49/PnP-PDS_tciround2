import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_graph_evolution():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 16
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = 'black'
    plt.rcParams["legend.handletextpad"] = 3.
    plt.rcParams["legend.markerscale"] = 1
    plt.rcParams["legend.borderaxespad"] = 0.
    plt.rcParams["figure.figsize"] =  (18, 7)

    filename_list = [
        './result/TCI-round2-poisson-2025-09-29_21-32-56/blur_blur_1_poisson1/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.002/data.npy',
        './result/TCI-round2-poisson-2025-09-29_21-32-56/blur_blur_1_poisson1/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.002/data.npy',
#        './result/TCI-round2-poisson-2025-09-29_21-32-56/random_sampling_poisson1/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
#        './result/TCI-round2-poisson-2025-09-29_21-32-56/random_sampling_poisson1/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',

#        './result/result-C-20240819/DATA_C-PnP-unstable-DnCNN_blur_00000_100_(03.png)_alpha10000_lambda80000.npy',
#        './result/result-C-20241119(proposed-revise)/DATA_C-Proposed_blur_00000_100_(03.png)_alpha10000_lambda0.00125.npy',
    ]
    data_list = [np.load(fn, allow_pickle=True).item() for fn in filename_list]

    # 各カラム=各メソッド
    method_list = [
        'PnP-PDS (NoBox)',
        'Proposed',
#        'Proposed',
#        'PnP-PDS (Unstable)',
    ]
    # メソッドごとに色を固定
    color_list = [
        '#E6AB02',  # PnP-PDS (Unstable)
        '#D00',  # Proposed
    ]

    fig, axes = plt.subplots(2, len(filename_list), tight_layout=True)

    for col, (method_data, method_name, color) in enumerate(zip(data_list, method_list, color_list)):
        first_curve_plotted = False
        for each_data in method_data['results'].values():
            # 上段: PSNR
            y_psnr = each_data['PSNR_evolution']
            axes[0, col].plot(
                y_psnr,
                color=color,
                linewidth=1.2,
                label=(method_name if not first_curve_plotted else '_nolegend_')
            )
            # 下段: c_n
            y_c = each_data['c_evolution']
            #y_c = each_data['other_data']['out_of_range_ratio']
            axes[1, col].plot(
                y_c,
                color=color,
                linewidth=1.2,
                label=(method_name if not first_curve_plotted else '_nolegend_')
            )
            first_curve_plotted = True

        # 体裁
        axes[0, col].set_title(method_name +  "  /  Deblurring")
        axes[0, col].set_ylim(1, 35)
        axes[0, col].set_ylabel("PSNR")
        axes[0, col].grid(color="gainsboro")
        axes[0, col].set_xlabel("iteration $n$")
        # グラフ内に凡例（右上）
        axes[0, col].legend(
            loc='upper right',
            frameon=True, facecolor='white', edgecolor='black',
            framealpha=0.95, handletextpad=0.7
        )

        axes[1, col].set_ylim(1e-7, 1e-1)
        #axes[1, col].set_ylim(1e-5, 1)
        axes[1, col].set_yscale('log')
        axes[1, col].set_ylabel(r"$c_n$")
        axes[1, col].grid(color="gainsboro")
        axes[1, col].set_xlabel("iteration $n$")
        axes[1, col].legend(
            loc='upper right',
            frameon=True, facecolor='white', edgecolor='black',
            framealpha=0.95, handletextpad=0.7
        )

    fig.tight_layout()
    plt.show()

    # 保存（必要ならパスは調整）
    fig.savefig('./result/TCI-round2-poisson-2025-09-29_21-32-56/graph_poisson_evolution-b.png', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('./result/TCI-round2-poisson-2025-09-29_21-32-56/graph_poisson_evolution-b.eps', bbox_inches="tight", pad_inches=0.05)

if (__name__ == '__main__'):
    plot_graph_evolution()
