import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

replacement_map = {
    "reply_DnCNN_nobn_nch_3_nlev_0.0075_dict": r"$\sigma_J=0.0075$",
    "reply_DnCNN_nobn_nch_3_nlev_0.01_dict": r"$\sigma_J=0.01$",
    "reply_DnCNN_nobn_nch_3_nlev_0.05_dict": r"$\sigma_J=0.05$",
    "reply_DnCNN_nobn_nch_3_nlev_0.1_dict": r"$\sigma_J=0.1$",
    "reply_DnCNN_nobn_nch_3_nlev_mixed0.1_dict": r"$\sigma_J\in[0,\,0.1]$"
}
def apply_hatch(ax, pats):
    for i, cont in enumerate(ax.containers):
        h = pats[i % len(pats)]
        for p in cont.patches:
            p.set_hatch(h)
            
def extract_sigmaJ_label(name): 
    return next((v for k, v in replacement_map.items() if k in name), "unknown")

def prepare_pivot(df, content):
    p = df.pivot_table(index='Gaussian_noise', columns='sigmaJ', values=content, aggfunc='max').reset_index()
    p.columns.name = None
    return p.astype({c: "float" for c in p.columns if c != "Gaussian_noise"})

def add_bar_edges(ax, color="#444", width=0.8):
    for container in ax.containers:
        for patch in container.patches:
            patch.set_edgecolor(color)
            patch.set_linewidth(width)

def plot_graph():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 18,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1.2,
        'legend.fancybox': False, 'legend.framealpha': 1, 'legend.edgecolor': '#CCC',
        'legend.handletextpad': 3., 'legend.markerscale': 1, 'legend.borderaxespad': 0.,
        'figure.figsize': (18, 7)
    })

    file_path = "./result/result-TCI-reply-discussion1/Gaussian/SUMMARY(20250604 100701 665848).txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    desired = ["Observation", "Gaussian_noise", "denoiser(architecture)", "PSNR", "SSIM"]
    cols = [i for i, c in enumerate(header) if c.strip() in desired]
    df = pd.read_csv(file_path, usecols=cols, skiprows=1, header=None, names=[header[i] for i in cols]).iloc[:-1]
    df["sigmaJ"] = df["denoiser(architecture)"].apply(extract_sigmaJ_label)
    df = df[~df["Gaussian_noise"].str.contains("0.0025|0.25")]
    df_blur, df_rs = df[df["Observation"] == "blur"], df[df["Observation"] == "random_sampling"]

    pivots = {f"{obs}_{m}": prepare_pivot(d, m) 
              for obs, d in zip(["blur", "rs"], [df_blur, df_rs]) for m in ["PSNR", "SSIM"]}

    fig, axes = plt.subplots(2, 2, figsize=(20, 7))
    colors = ["#7B89F0DD", "#9DDF77DD", "#ECDD7DDD", "#EA8686DD", '#777A']
    patterns =["//", "..", "xxx", "\\\\", ""] 
    barargs = dict(width=0.95, color=colors)

    # 上段：xラベル非表示
    pivots["blur_PSNR"].plot.bar(x='Gaussian_noise', ax=axes[0,0], title="Deblurring", ylabel="PSNR", xlabel="", **barargs)
    pivots["rs_PSNR"].plot.bar(x='Gaussian_noise', ax=axes[0,1], title="Inpainting", ylabel="PSNR", xlabel="", **barargs)
    # 下段：xラベル = σ
    pivots["blur_SSIM"].plot.bar(x='Gaussian_noise', ax=axes[1,0], xlabel=r"$\sigma$", ylabel="SSIM", **barargs)
    pivots["rs_SSIM"].plot.bar(x='Gaussian_noise', ax=axes[1,1], xlabel=r"$\sigma$", ylabel="SSIM", **barargs)

    # 各バーに枠線を追加
    for ax in axes.flat:
        add_bar_edges(ax, color="#555", width=0.8)
        ax.legend_.remove()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, axis='both', linewidth=0.5, color='gray', alpha=0.4)
        ax.set_axisbelow(True)
        apply_hatch(ax,patterns)
    
    axes[0,0].set_ylim(8,40); axes[0,1].set_ylim(8,40)
    axes[1,0].set_ylim(0,1);  axes[1,1].set_ylim(0,1)
    for a in axes[0]: a.tick_params(labelbottom=False)  # 上段のx目盛りラベル非表示

    left, right = min(ax.get_position().x0 for ax in axes.flat), max(ax.get_position().x1 for ax in axes.flat)
    h, l = axes[0,0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(left,0.09,right-left,0.05), mode="expand", ncol=len(l))
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    for ext in ["png", "pdf", "eps"]:
        fig.savefig(f'./result/result-TCI-reply-discussion1/Gaussian/sigmaJ_performance.{ext}', 
                    bbox_inches="tight", pad_inches=0.05)

if __name__ == '__main__':
    plot_graph()
