import os
import numpy as np
import cv2


# ========= 設定 =========
#IMG_NUM = 0
#NP_FILE = "./result/2025-10-21_18-16-08/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.0015/data.npy"
#OUT_DIR = "./result/2025-10-21_18-16-08/blur_blur_1_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.0015/export"
#ITER_LIST = range(0, 1501, 30)   # None=全フレーム

IMG_NUM = 1
NP_FILE = "./result/2025-10-21_18-16-08/random_sampling_0.3_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.001/data.npy"
OUT_DIR = "./result/2025-10-21_18-16-08/random_sampling_0.3_poisson10/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.001/export"
ITER_LIST = range(0, 3001, 60)   # None=全フレーム

#IMG_NUM = 2
#NP_FILE = "./result/2025-10-21_18-16-08/random_sampling_0.3_poisson50/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00075/data.npy"
#OUT_DIR = "./result/2025-10-21_18-16-08/random_sampling_0.3_poisson50/methods/C-Proposed_[DnCNN_nobn_nch_1_nlev_0.01_journal/lamb_0.00075/export"
#ITER_LIST = range(0, 3001, 60)   # None=全フレーム

TEXT_COLOR = (255, 255, 255)  # RGB形式 (例: 緑=(0,255,0), シアン=(0,255,255), 赤=(255,50,50))
FONT = cv2.FONT_HERSHEY_DUPLEX  # デジタルっぽいフォント
FONT_SCALE = 0.8
THICKNESS = 1
# ========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUT_DIR, "evol_cv_png")
    os.makedirs(img_dir, exist_ok=True)

    obj = np.load(NP_FILE, allow_pickle=True)
    data = obj if isinstance(obj, np.lib.npyio.NpzFile) else (obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj)

    entry = data["results"][IMG_NUM]
    other = entry["other_data"]

    # fne_data 保存
    fne = np.asarray(other.get("fne_data"))
    with open(os.path.join(OUT_DIR, f"fne_data_img{IMG_NUM}.txt"), "w", encoding="utf-8") as f:
        f.write(f"# fne_data for IMG_NUM={IMG_NUM}\n")
        for it, val in enumerate(fne):
            f.write(f"iter {it}: {val}\n")

    evol = np.asarray(other["evol_data"])
    psnr = np.asarray(entry.get("PSNR_evolution", []), dtype=float)
    max_iter = evol.shape[0]
    iters = range(max_iter) if ITER_LIST is None else [i for i in ITER_LIST if 0 <= i < max_iter]

    for it in iters:
        pic = evol[it]
        if pic.ndim == 3:
            pic = np.moveaxis(pic, 0, 2)  # CHW→HWC
        pic = np.clip(pic, 0.0, 1.0)
        img8 = (pic * 255.0).astype(np.uint8)
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        val = psnr[it] if it < len(psnr) and np.isfinite(psnr[it]) else None
        txt = f"iter" if val is not None else f"iter {it} | PSNR: N/A"
        cv2.putText(img8, txt, (12, 213), FONT, FONT_SCALE, TEXT_COLOR[::-1], THICKNESS, cv2.LINE_AA)

        txt = f"PSNR:" if val is not None else f"iter {it} | PSNR: N/A"
        cv2.putText(img8, txt, (12, 243), FONT, FONT_SCALE, TEXT_COLOR[::-1], THICKNESS, cv2.LINE_AA)

        txt = f"{it}" if val is not None else f"iter {it} | PSNR: N/A"
        cv2.putText(img8, txt, (100, 213), FONT, FONT_SCALE, TEXT_COLOR[::-1], THICKNESS, cv2.LINE_AA)

        txt = f"{val:.2f} dB" if val is not None else f"iter {it} | PSNR: N/A"
        cv2.putText(img8, txt, (100, 243), FONT, FONT_SCALE, TEXT_COLOR[::-1], THICKNESS, cv2.LINE_AA)

        # ↑ OpenCVはBGRなので TEXT_COLOR[::-1] でRGB→BGR変換

        cv2.imwrite(os.path.join(img_dir, f"evol_iter_{it:06d}.png"), img8)

    print(f"[OK] {len(iters)} フレームを書き出しました -> {img_dir}")

if __name__ == "__main__":
    main()
