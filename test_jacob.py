import numpy as np
import operators as op
import glob, os, cv2, torch
import json
import utils.utils_jacobian as utils_jacobian

from models.denoiser import Denoiser as Denoiser_J
from models.network_dncnn import DnCNN as Denoiser_KAIR
from utils.utils_eval import eval_psnr, eval_ssim
from algorithm.admm import *

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

with open('config/setup.json', 'r') as f:
    config = json.load(f)

def compute_reg(data, model, reg_fun):
    """
    Computes the regularization reg_fun applied to the correct point
    """
    torch.cuda.empty_cache()

    im = torch.from_numpy(data).unsqueeze(0).type(Tensor)
    #im = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).type(Tensor)
    data_torch = im.requires_grad_()
    out_torch = model.denoise(data_torch)
    out_q_torch = 2 * out_torch - data_torch

    jac_norm = reg_fun(data_torch, out_q_torch)
    return jac_norm.item()

def evol_jacob(architecture):
    path_prox = config['root_folder'] + 'nn/' + architecture + '.pth'
    denoiser_J = Denoiser_J(file_name=path_prox, ch=3)
    reg_fun = utils_jacobian.JacobianReg_l2(eval_mode=True, max_iter=10, tol=1e-1)

    pattern_red = '*.npy'
    path_test = 'C:/Users/temp/Documents/lab/PnP-PDS/result/result-TCI-round2-Jacob/obsrv_001_blur_wo_clipping_iter10_tol1e-1/'
    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))

    output_path = os.path.join(path_test, 'jacobian_values.txt')

    with open(output_path, 'w') as f_out:
        for path_img in path_images:
            ext = os.path.splitext(path_img)[1].lower()
            
            if ext == ".npy":
                # npyファイルをロード
                x = np.load(path_img)
                if x.dtype != np.float32:
                    x = x.astype(np.float32)
            else:
                # 画像ファイルをロード
                x = cv2.imread(path_img)
                x = np.asarray(x, dtype="float32") / 255.
                x = np.clip(x, 0, 1)
                x = np.moveaxis(x, -1, 0)

            jac_norm = compute_reg(x, denoiser_J, reg_fun)
            img_name = os.path.basename(path_img)

            # 標準出力とファイル出力
            print(f"{img_name}: Jacobian norm = {jac_norm}")
            f_out.write(f"{img_name}: {jac_norm:.6f}\n")

if __name__ == "__main__":
    evol_jacob('DnCNN_nobn_nch_3_nlev_0.01_journal')