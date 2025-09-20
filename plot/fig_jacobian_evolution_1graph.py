import matplotlib.pyplot as plt
import numpy as np

def plot_graph_evolution():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 14
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = 'black'
    plt.rcParams["legend.handletextpad"] = 3.
    plt.rcParams["legend.markerscale"] = 1
    plt.rcParams["legend.borderaxespad"] = 0.
    plt.rcParams["figure.figsize"] = (16, 12)

    filename_list = [ # blur at sigma=0.01
        './result/result-TCI-reply-instability-all/DATA_A-PnPPDS-DnCNN-wo-constraint_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-RED-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-PnPFBS-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-Proposed_DnCNN_nobn_nch_3_nlev_0.01_journal_blur_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
    ]
    filename_list = [
        './result/result-TCI-reply-instability-all/DATA_A-PnPPDS-DnCNN-wo-constraint_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-RED-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-PnPFBS-DnCNN_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
        './result/result-TCI-reply-instability-all/DATA_A-Proposed_DnCNN_nobn_nch_3_nlev_0.01_journal_random_sampling_0.010_300_(ILSVRC2012_val_00044012.JPEG.png)_alpha0.900_lambda10000.npy',
    ]

    method_list = [
        'PnP-PDS (w/o a box const.)',
        'RED',
        'PnP-FBS',
        'Proposed',
    ]
    plotColor = [
        "#E6AB02",
        "#1B9E77",
        "#00D",
        "#D00",
    ]

    # データ読み込み
    all_data = {}
    max_imgs = 0
    for idx, filename in enumerate(filename_list):
        file = np.load(filename, allow_pickle=True).item()
        results = file['results']
        fne_list = [v['other_data']['fne_data'] for v in results.values()]
        all_data[method_list[idx]] = fne_list
        max_imgs = max(max_imgs, len(fne_list))

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6), sharey=True)
    axes = axes.flatten()

    for img_idx in range(nrows * ncols):
        ax = axes[img_idx]
        if img_idx < max_imgs:
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
            for m_idx, method in enumerate(method_list):
                try:
                    y = all_data[method][img_idx]
                    ax.plot(y, color=plotColor[m_idx], linewidth=1.0)
                except IndexError:
                    continue
            ax.set_title(f"Image {img_idx + 1}")
            ax.set_xlabel("iteration $n$")
            ax.grid(color="gainsboro")
            #ax.set_yscale('log')
            #ax.set_ylim(1e-1, 1e10)
            ax.set_ylim(0, 10)
            if img_idx % ncols == 0:
                ax.set_ylabel("$\\|\\nabla Q(\mathbf{x}_n)\\|_{\mathrm{sp}}^2$")
        else:
            # 凡例用の空欄領域
            handles = [plt.Line2D([0], [0], color=plotColor[i], label=method_list[i], linewidth=2.0)
                       for i in range(len(method_list))]
            ax.legend(handles=handles, loc='center')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.show()

    fig.savefig('./result/result-TCI-reply-instability/Jacob_Q_evol_deb_proposed.png', bbox_inches="tight", pad_inches=0.1)
    fig.savefig('./result/result-TCI-reply-instability/Jacob_Q_evol_deb_proposed.eps', bbox_inches="tight", pad_inches=0.1)


if __name__ == '__main__':
    plot_graph_evolution()
