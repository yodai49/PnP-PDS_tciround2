import numpy as np

def compute_out_of_range_ratio():
    filename_list = [
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/blur_blur_1_poisson1/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.002/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/blur_blur_1_poisson1/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.002/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/blur_blur_1_poisson2/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.002/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/blur_blur_1_poisson2/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.002/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.0015/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/blur_blur_1_poisson10/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.0015/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/random_sampling_poisson1/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/random_sampling_poisson1/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/random_sampling_poisson2/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/random_sampling_poisson2/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00125/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/random_sampling_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.001/data.npy',
        './result/TCI-round2-poisson-main 2025-09-22_20-14-05/random_sampling_poisson10/methods/C-PnPPDS-DnCNN-wo-constraint_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.001/data.npy',
    ]
    filename_list = [
        './result/2025-09-29_21-32-56/blur_blur_1_poisson1/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.001/data.npy',
    ]
    for fn in filename_list:
        data = np.load(fn, allow_pickle=True).item()

        ratios = []
        for img_name, each_data in data['results'].items():
            arr = np.array(each_data['RESULT'])
            total = arr.size
            #print(arr)
            #print(np.min(arr), np.max(arr))
            ratio = 1 - np.sum((arr >= 0) & (arr <= 1) & (~np.isnan(arr))) / arr.size
            #ratio = out_of_range / total if total > 0 else 0.0
            ratios.append((img_name, ratio))
            print(each_data['other_data']['out_of_range_ratio'][0])

        avg_ratio = np.mean([r for _, r in ratios]) if ratios else 0.0

        # 標準出力（一行形式）
        inline = f"File: {fn} | " + " ".join([f"{img}:{val:.6f}" for img, val in ratios])
        inline += f" | Average:{avg_ratio:.6f}"
        print(inline)
if __name__ == "__main__":
    compute_out_of_range_ratio()
