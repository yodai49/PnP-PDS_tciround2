import numpy as np
import operators as op
import math
import time, torch
import utils.utils_jacobian as utils_jacobian

from models.denoiser import Denoiser as Denoiser_J
from models.network_dncnn import DnCNN as Denoiser_KAIR
from utils.utils_eval import eval_psnr, eval_ssim
from algorithm.admm import *
#from DPIR.utils import utils_pnp as pnp
#from DPIR.utils import utils_model

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_reg(data, model, reg_fun):
    """
    Computes the regularization reg_fun applied to the correct point
    """
    torch.cuda.empty_cache()  # Commented out for realsn

    im = torch.from_numpy(data).unsqueeze(0).type(Tensor)  # Moving to torch
    data_torch = im.requires_grad_()  # Activate requires_grad for backprop
    out_torch = model.denoise(data_torch)
    out_q_torch = 2*out_torch-data_torch

    jac_norm = reg_fun(data_torch, out_q_torch)

    return jac_norm.item()

def test_iter(x_0, x_obsrv, x_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2, gaussian_nl, sp_nl, poisson_alpha, path_prox, max_iter, method="A-Proposed", ch = 3, r=1, DRUNet_solver = None):
    # x_0　     初期値
    # x_obsrv   観測画像
    # x_true    真の画像
    # phi, adj_phi 観測作用素とその随伴作用素
    # gamma1, gamma2 PDSのステップサイズ
    # alpha_s   スパースノイズのalpha
    # alpha_n   ガウシアンノイズのalpha
    # myLambda  PnP-FBSのステップサイズ
    # gaussian_nl, sp_nl　ガウシアンノイズの分散とスパースノイズの重畳率
    # path_prox ガウシアンデノイザーのパス
    # max_iter アルゴリズムのイタレーション数
    # method 手法
    x_n = x_0
    y_n = np.zeros(x_0.shape) # 次元が画像と同じ双対変数
    y1_n = np.concatenate([np.zeros(x_0.shape), np.zeros(x_0.shape)], 0)
    y2_n = np.zeros(x_0.shape)
    s_n = np.zeros(x_0.shape)
    z_n = np.zeros(x_0.shape)
    d_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    ssim_data = np.zeros(max_iter)
    evol_data = np.zeros(x_0.shape)
    fne_data = np.zeros(max_iter)
    y1_evol = np.zeros(max_iter)
    y2_evol = np.zeros(max_iter)
    y1_val_evol = np.zeros(max_iter)
    out_of_range_ratio = np.zeros(max_iter)
    reg_fun = utils_jacobian.JacobianReg_l2(eval_mode=True, max_iter=10, tol=1e-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sigma_J = 0.01
    sigma = 0.01
    n = x_obsrv - phi(x_true)

    if (method.find('unstable') != -1):
        if (ch == 3):
            nb = 20
        elif(ch == 1):
            nb = 17
        denoiser_KAIR = Denoiser_KAIR(in_nc=ch, out_nc=ch, nc=64, nb=nb, act_mode='R', model_path = path_prox)
    elif(method.find('Proposed') != -1 or method.find('DnCNN') != -1):
        denoiser_J = Denoiser_J(file_name=path_prox, ch = ch)
    elif (method.find('DRUNet') != -1):
        model_path = path_prox
        from DPIR.models.network_unet import UNetRes as net
        model = net(in_nc=ch+1, out_nc=ch, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    
        rhos, sigmas = pnp.get_rho_sigma(sigma=sigma, iter_num = max_iter, modelSigma1=49, modelSigma2 = sigma * 255, w=1) 
        sigmas = torch.tensor(sigmas).to(device)
        rhos = torch.tensor(rhos).to(device)
        
    totaltime = 0
    for i in range(max_iter):
        start_time = time.process_time()
        x_prev = x_n
        s_prev = s_n

        if(method == 'A-Proposed'):
            # Primal-dual spilitting algorithm with denoiser (Gaussian noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)

#            Without hard constraint
#            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
#            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
#            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'B-Proposed'):
            # Primal-dual spilitting algorithm with denoiser  (Gaussian noise + sparse noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl, r)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        elif(method == 'C-Proposed'):
            # Primal-dual spilitting algorithm with denoiser (Poisson noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)


#            Without box constraint
#            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
#            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
#            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)



        ########################################################
        #   For Experiment A (Gaussian noise )                 #
        ########################################################

        elif(method == 'A-PnPFBS-DnCNN'):
            # Forward-backward spilitting algorithm with DnCNN
            x_n = denoiser_J.denoise(x_n - gamma1 * myLambda * 0.5 * (op.grad_x_l2(x_n, np.zeros(x_n.shape), phi, adj_phi, x_obsrv)))
        elif(method == 'A-PnPPDS-BM3D'):
            # BM3D-PnP-PDS (constrained formulation)
            x_n = x_n - gamma1 *adj_phi(y_n)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=np.sqrt(gamma1))
            x_n = np.moveaxis(x_n, -1, 0)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        elif(method == 'A-PnPFBS-BM3D'):
            # BM3D-PnP-FBS (additive formulation)
            x_n = x_n - gamma1 * op.grad_x_l2(x_n, np.zeros(x_n.shape) , phi, adj_phi, x_obsrv)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=1)
            x_n = np.moveaxis(x_n, -1, 0)       
        elif(method == 'A-PDS-TV'):
            # Primal-dual spilitting algorithm with TV
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev))
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        elif(method == 'A-FBS-TV'):
            # Primal-dual splitting with TV (additive formulation):
            x_n = x_n - gamma1 * (adj_phi(phi(x_n)-x_obsrv) + op.D_T(y1_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)   
        elif(method == 'A-RED-DnCNN'):
            # DnCNN RED 
            # https://arxiv.org/pdf/1611.02862.pdf のsigmaをgamma1にlambdaをmyLambdaに置き換えた
            x_n = denoiser_J.denoise(x_n)
            mu = 2 / (1/gamma1**2 + myLambda)
            x_n = x_prev - mu * ((1 / gamma1**2) * adj_phi(phi(x_prev) - x_obsrv) + myLambda * (x_prev - x_n))
        elif(method == 'A-PnPPDS-unstable-DnCNN'):
            x_n = x_n - gamma1 * (adj_phi(y_n) + y2_n)
            x_n_tensor = torch.from_numpy(np.ascontiguousarray(x_n)).float().unsqueeze(0)
            x_n_tensor = denoiser_KAIR(x_n_tensor)
            x_n = x_n_tensor.data.squeeze().detach().numpy().copy()
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)
        elif (method=='A-PnPPDS-DnCNN-wo-constraint'):
            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        elif (method=='A-PnPPDS-DnCNN-clipping-layer'):
            x_n = np.clip(denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n)), 0, 1)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        elif (method == 'A-PnPPDS-DRUNet'):
            # x_n=[1, C, H, W]
            x_n = torch.from_numpy(x_n).unsqueeze(0).to(device).float()
            x_n = torch.cat((x_n, sigmas[i].float().repeat(1, 1, x_n.shape[2], x_n.shape[3])), dim=1)
            # x_n=[1, C + 1, H, W]
            x_n = utils_model.test_mode(model, x_n, mode=2, refield=32, min_size=128, modulo=16)
            x_n = x_n.squeeze(0).cpu().numpy()
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)
        elif(method == 'A-PnPFBS-DRUNet'):
            # Forward-backward spilitting algorithm with DnCNN
            #x_n = x_n - gamma1 * myLambda * 0.5 * (op.grad_x_l2(x_n, np.zeros(x_n.shape), phi, adj_phi, x_obsrv))
            x_prior = torch.from_numpy(x_n).unsqueeze(0).to(device).float()
            x_prior = torch.cat((x_prior, sigmas[i].float().repeat(1, 1, x_prior.shape[2], x_prior.shape[3])), dim=1)
            # x_n=[1, C + 1, H, W]
            x_prior = utils_model.test_mode(model, x_prior, mode=2, refield=32, min_size=128, modulo=16)
            x_prior = x_prior.squeeze(0).cpu().numpy()
            x_n = DRUNet_solver(x_prior, x_obsrv, alpha= myLambda * sigma * sigma / (sigmas[i].item() * sigmas[i].item()))
            #x_n = data_solution_grad_descent(x_prior, x_obsrv, phi, adj_phi, myLambda * sigma * sigma / (sigmas[i].item() * sigmas[i].item()))

        ########################################################
        #   For Experiment B (Sparse noise + Gaussian noise )  #
        ########################################################
        elif(method == 'comparisonB-1'):
            # BM3D-PnP-PDS (Gaussian noise + sparse noise)
            x_n = x_n - gamma1 * adj_phi(y_n)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=0.01)
            x_n = np.moveaxis(x_n, -1, 0)
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'comparisonB-2'):
            # ADMM algorithm with denoiser  (Gaussian noise + sparse noise)
            x_n = step1ofADMMforSparseX(s_n, z_n, y_n, phi, adj_phi, path_prox, ch, gamma1, m1)
            s_n = step1ofADMMforSparseS(x_n, z_n, y_n, phi, alpha_s, sp_nl, gamma1, m2)
            z_n = op.proj_l2_ball(phi(x_n) + s_n + y_n, alpha_n, gaussian_nl, sp_nl, x_obsrv)
            y_n = y_n + phi(x_n) + s_n - z_n
        elif(method == 'comparisonB-3'):
            # HTV (constrained formulation)
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            s_n = op.proj_l1_ball(s_n - gamma1 * y2_n, alpha_s, sp_nl)
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'comparisonB-4'):
            # DnCNN RED 
            # https://arxiv.org/pdf/1611.02862.pdf のsigmaをgamma1にlambdaをmyLambdaに置き換えた
            x_n = x_prev - gamma1 * (myLambda * adj_phi(phi(x_prev) + s_n - x_obsrv) + (x_prev - denoiser_J.denoise(x_n)))
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_obsrv)), alpha_s, sp_nl)
        elif(method == 'comparisonB-5'):
            # DnCNN-PnP-FBS (additive formulation)
            x_n = denoiser_J.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, s_n, phi, adj_phi, x_obsrv)))
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_obsrv)), alpha_s, sp_nl)


        ########################################################
        #   For Experiment C (Poisson noise )                  #
        ########################################################

        elif(method == 'C-PnPPDS-BM3D'):
            # BM3D-PnP-PDS (Poisson noise) 
            x_n = bm3d.bm3d(x_n - gamma1 * adj_phi(y_n), sigma_psd = np.sqrt(gamma1))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
        elif(method == 'C-PnPADMM-DnCNN'):
            # DnCNN-PnP-ADMM (Poisson noise)
            x_n = step1ofADMMforPoisson (d_n, z_n, x_obsrv, phi, adj_phi, poisson_alpha, myLambda, m1, gammaInADMMStep1)
            z_n = denoiser_J.denoise(x_n + d_n)
            d_n = d_n + x_n - z_n
        elif(method == 'C-RED-DnCNN'):
            # DnCNN RED (Poisson noise)
            x_str = denoiser_J.denoise(x_n)
            x_n = x_n - gamma1 * (x_n - x_str + adj_phi(y_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
        elif(method == 'C-PnP-unstable-DnCNN'):
            # Not firmly-nonexpansive DnCNN (Poisson noise)
            x_n = x_n - gamma1 * (adj_phi(y_n) + y2_n)
            x_n_tensor = torch.from_numpy(np.ascontiguousarray(x_n)).float().unsqueeze(0).unsqueeze(0)
            x_n_tensor = denoiser_KAIR(x_n_tensor)
            x_n = x_n_tensor.data.squeeze().squeeze().detach().numpy().copy()

            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)
        elif (method == 'C-PnPPDS-DnCNN-wo-constraint'):
             x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
             y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
             y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
        elif (method == 'C-PnPPDS-DnCNN-clipping-layer'):
            # Primal-dual spilitting algorithm with denoiser (Poisson noise)
            x_n = np.clip(denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n)), 0, 1)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)


        else:
            print("Unknown method:", method)
            return x_n, s_n+0.5, c, psnr_data, ssim_data, average_time

        torch.cuda.synchronize(); 
        end_time = time.process_time()
        totaltime+=end_time-start_time

        c[i] = np.linalg.norm((x_n - x_prev).flatten(), 2) / np.linalg.norm(x_prev.flatten(), 2)
        if (c[i] < -1):
            print("Convergence detected at iteration", i)
            psnr_data[i:] = eval_psnr(x_true, x_n)
            ssim_data[i:] = eval_ssim(x_true, x_n)
            break


        psnr_data[i] = eval_psnr(x_true, x_n)
        ssim_data[i] = eval_ssim(x_true, x_n)
        if False and ((i % 10 == 0) or (i == 0) or (i == max_iter - 1)):
            print(f"[{method}] iter {i+1:4d}/{max_iter:4d} | "
                  f"PSNR={psnr_data[i]:.2f} dB | SSIM={ssim_data[i]:.4f} | "
                  f"rel_change={c[i]:.3e} | "
                  f"out_of_range={out_of_range_ratio[i]*100:.2f}%")
#        evol_data = np.append(evol_data, x_n)  # iterationごとの変化を追いたい場合には入れる
#        if(method.find('Proposed') != -1):
#            const = i * 2 / max_iter
#            x1 = np.random.rand(*x_n.shape) * const
#            x2 = np.random.rand(*x_n.shape) * const
#            x1 = x_n
#            x2 = x_prev
#            x1_denoise = denoiser_J.denoise(x1)
#            x2_denoise = denoiser_J.denoise(x2)
#            l2 = np.linalg.norm(x1_denoise.flatten() - x2_denoise.flatten(), 2)**2
#            ip = np.dot((x1 - x2).flatten(), (x1_denoise - x2_denoise).flatten())
#            fne_data[i] = l2 / ip
        out_of_range_ratio[i] = 1 - np.sum((x_n >= 0) & (x_n <= 1) & (~np.isnan(x_n))) / x_n.size
        y1_evol[i] = np.linalg.norm(y_n.flatten(), 2)
        y2_evol[i] = np.linalg.norm(y2_n.flatten(), 2)
        y1_val_evol[i] = np.linalg.norm((gamma1 * y_n + sigma_J * n / sigma).flatten(), 2)

        #if np.sum(np.isnan(x_n)) > 0:
        #    fne_data[i] = math.nan
        #else:
        #    fne_data[i] = compute_reg(x_n, denoiser_J, reg_fun)

    average_time = totaltime/max_iter

    others_data = {}
    others_data['evol_data'] = evol_data
    others_data['fne_data'] = fne_data
    others_data['out_of_range_ratio'] = out_of_range_ratio
    others_data['y1'] = y_n
    others_data['y2'] = y2_n
    others_data['y1_evol'] = y1_evol
    others_data['y2_evol'] = y2_evol
    others_data['y1_val_evol'] = y1_val_evol
    others_data['n'] = n

    return x_n, s_n+0.5, c, psnr_data, ssim_data, average_time, others_data







def data_solution_grad_descent(x_prior, y, phi, adj_phi, alpha=0.01, 
                                 num_iter=30, verbose=False):
    """
    Solve (1/2)||Ax - y||^2 + (alpha/2)||x - x_prior||^2 via gradient descent.

    Args:
        x_prior (Tensor): prior image estimate, shape [1, C, H, W]
        y (Tensor): observed data, same shape as A(x), typically [1, C, H, W]
        phi (function): forward operator A(x)
        adj_phi (function): adjoint operator A^T(x)
        alpha (float): regularization parameter
        lr (float): gradient descent step size
        num_iter (int): number of inner iterations
        verbose (bool): whether to print residuals

    Returns:
        Tensor: estimated image x, shape [1, C, H, W]
    """
    x = x_prior.copy()
    lr = 1 / (1 + alpha)  # Step size for gradient descent

    for i in range(num_iter):
        Ax_minus_y = phi(x) - y
        grad_data = adj_phi(Ax_minus_y)
        grad_prior = alpha * (x - x_prior)
        grad = grad_data + grad_prior

        x = x - lr * grad

        if verbose and (i % 1 == 0 or i == num_iter - 1):
            loss_data = 0.5 * np.sum((Ax_minus_y) ** 2)
            loss_prior = 0.5 * alpha * np.sum((x - x_prior) ** 2)
            print(f"[Iter {i+1}] Loss: {loss_data + loss_prior:.6e}")

    return x
