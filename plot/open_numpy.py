import os
import csv
import numpy as np
import cv2


# ========= 設定 =========
NP_FILE = "./result/2025-10-21_18-16-08/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.0015/data.npy"
#NP_FILE = "./result/2025-10-02_21-49-29/random_sampling_gaussian0.01/methods/A-Proposed_[DnCNN_nobn_nch_3_nlev_0.01_journal/alpha_1/data.npy"
IMG_NUM = 0
OUT_DIR = "./result/2025-10-21_18-16-08/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.0015/export"
#OUT_DIR = "./result/2025-10-02_21-49-29/random_sampling_gaussian0.01/methods/A-Proposed_[DnCNN_nobn_nch_3_nlev_0.01_journal/export"
FNE_SAVE_BASENAME = "fne_data"
ITER_LIST = range(0, 500, 10)   # None=全フレーム / 例: [0,10,20] で指定
AXES_HINT = None   # None=自動推定 / "CHW" or "HWC" or "HW"
# 正規化モード: "percentile"(推奨), "01"(0..1を255倍), "m11"([-1,1]を0..255), "none"(クリップのみ)
NORMALIZE = "percentile"
P_LOW, P_HIGH = 1.0, 99.0    # percentileの下/上限（NORMALIZE="percentile"の時のみ有効）
# ========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUT_DIR, "evol_cv_png")
    os.makedirs(img_dir, exist_ok=True)

    obj = np.load(NP_FILE, allow_pickle=True)
    data = obj if isinstance(obj, np.lib.npyio.NpzFile) else (obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj)

    entry = data["results"][IMG_NUM]
    other = entry["other_data"]

    # fne_data 保存（おまけ）
    fne_data = np.asarray(other.get("fne_data"))
    outfile = os.path.join(OUT_DIR, f"fne_data_img{IMG_NUM}.txt")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"# fne_data for IMG_NUM={IMG_NUM}\n")
        for it, val in enumerate(fne_data):
            f.write(f"iter {it}: {val}\n")

    evol = np.asarray(other["evol_data"])       # 形: (max_iter, ...) を想定
    max_iter = evol.shape[0]
    frame_shape = evol.shape[1:]

    # 出力フレーム集合
    if ITER_LIST is None:
        iter_range = range(max_iter)
    else:
        iter_range = [i for i in ITER_LIST if 0 <= i < max_iter]

    # --- 画像保存（OpenCVはBGR想定）---
    for it in iter_range:
        picture = evol[it]
        picture_temp = picture
        if(np.ndim(picture) == 3):
            # color
            picture_temp = np.moveaxis(picture, 0, 2)
        picture_temp[picture_temp > 1.] = 1.
        picture_temp[picture_temp < 0.] = 0.

        cv2.imwrite(os.path.join(img_dir, f"evol_iter_{it:06d}.png") , np.uint8(picture_temp*255.))
        
    print(f"[OK] {len(iter_range)} フレームを書き出しました -> {img_dir}")

if __name__ == "__main__":
    main()
