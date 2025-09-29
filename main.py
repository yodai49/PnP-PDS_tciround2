import cv2, datetime, glob, json, os, iteration
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from operators import get_observation_operators
from utils.utils_image import save_imgs
from utils.utils_noise import add_salt_and_pepper_noise, add_gaussian_noise, apply_poisson_noise
from utils.utils_eval import eval_psnr, eval_ssim
from utils.utils_parse_args import *
from utils.utils_unparse_args import *
from DPIR.solver import make_solver
from utils.utils_method_master import get_algorithm_denoiser
from utils.utils_textfile import *

with open('config/setup.json', 'r') as f:
    config = json.load(f)

def test_all_images (experimental_settings_arg = {}, method_arg = {}, configs_arg = {}, paths = {}):
    gaussian_nl, sp_nl, poisson_noise, poisson_alpha, deg_op, r, blur_kernel  = parse_args_exp (experimental_settings_arg)
    method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2 = parse_args_method (method_arg)
    ch, add_timestamp, result_output = parse_args_configs (configs_arg)
    experimental_settings_all = unparse_args_exp (gaussian_nl, sp_nl, poisson_noise, poisson_alpha, deg_op, r, blur_kernel)
    method_all = unparse_args_method (method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2)
    configs_all = unparse_args_configs (ch, add_timestamp, result_output)

    pattern_red = config['pattern_red']
    path_test = config['path_test']
    path_kernel = config['root_folder'] + 'blur_models/' + blur_kernel + '.mat'
    path_prox = config['root_folder'] + 'nn/' + architecture + '.pth'
    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))
    path_result = paths['result']
    path_observation = paths['observation']
    path_groundtruth = paths['groundtruth']

    psnr = np.zeros((len(path_images)))
    ssim = np.zeros((len(path_images)))
    cpu_time = np.zeros((len(path_images)))
    results = {}

    deg_op_supp = ''
    if(deg_op == 'random_sampling'):
        deg_op_supp = '_r' + str(r).ljust(3, '0')
    elif(deg_op == 'blur'):
        deg_op_supp = '_' + blur_kernel
    path_base = method + '_' + architecture + '_' + deg_op +  deg_op_supp +  '_' + str(gaussian_nl).ljust(5, '0') + '_' + str(poisson_alpha)

    for path_img in path_images:
        # =====================================
        # Prepare images and operators
        # =====================================
        index = path_images.index(path_img)
        img_true = cv2.imread(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        if(ch == 1):
            # Gray scale
            img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        elif(ch == 3):
            # Color  (3 x H x W)
            img_true = np.moveaxis(img_true, -1, 0)
        phi, adj_phi = get_observation_operators(operator = deg_op, path_kernel = path_kernel, r = r)
        Id, _ = get_observation_operators("Id", path_kernel, r)
        img_obsrv = phi(img_true)
        if(deg_op == 'blur' or deg_op == 'Id'):
            img_obsrv = add_gaussian_noise(img_obsrv, gaussian_nl, Id)
        elif (deg_op == 'random_sampling'):
            img_obsrv = add_gaussian_noise(img_obsrv, gaussian_nl, phi)        
        if(poisson_noise):
            img_obsrv = apply_poisson_noise(img_obsrv, poisson_alpha)
        if(deg_op == 'blur' or deg_op == 'Id'):
            img_obsrv = add_salt_and_pepper_noise(img_obsrv, sp_nl, Id)
        elif (deg_op == 'random_sampling'):
            img_obsrv = add_salt_and_pepper_noise(img_obsrv, sp_nl, phi)        
        x_0 = np.copy(img_obsrv)
        if(poisson_noise):
            x_0 = x_0 / poisson_alpha
        
        # =====================================
        # Run evaluation
        # =====================================
        if (method.find('DRUNet') != -1):
            h = scipy.io.loadmat(path_kernel)
            h = np.array(h['blur'])
            solver = make_solver(deg_op=deg_op, h=h, phi=phi, adj_phi=adj_phi)
        else:
            solver = None
        if (method == 'get_observation_npy'):
            img_obsrv = np.clip(img_obsrv, 0, 1)
            np.save(path_observation + '\OBSERVATION_' + path_base + (path_img[path_img.rfind('\\'):])[1:], img_obsrv)
            continue

        img_sol, s_sol, c_evolution, psnr_evolution, ssim_evolution, average_time, other_data = iteration.test_iter(x_0, img_obsrv, img_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2, gaussian_nl, sp_nl, poisson_alpha, path_prox, max_iter, method, ch, r, solver)
        if(poisson_noise):
            img_obsrv = img_obsrv / poisson_alpha

        # =====================================
        # Save results
        # =====================================
        filename = (path_img[path_img.rfind('\\'):])[1:]
        psnr[index] = psnr_evolution[-1]
        ssim[index] = ssim_evolution[-1]
        cpu_time[index] = average_time
        psnr_obsrv = eval_psnr(img_true, img_obsrv)
        ssim_obsrv = eval_ssim(img_true, img_obsrv)
        results[index] = {'filename' : filename, 'c_evolution': c_evolution, 'PSNR_evolution' : psnr_evolution, 'SSIM_evolution' : ssim_evolution, 'GROUND_TRUTH': img_true, 'OBSERVATION' : img_obsrv, 'RESULT': img_sol, 'REMOVED_SPARSE': s_sol, 'PSNR' : psnr_evolution[-1], 'SSIM' : ssim_evolution[-1], 'CPU_time' : average_time, 'PSNR_observation' : psnr_obsrv, 'SSIM_observation' : ssim_obsrv}
        results[index]['other_data'] = other_data

        # =====================================
        # Save images
        # =====================================
        pictures = [img_true, img_obsrv, img_sol]
        path_saveimg_base = filename
        if (add_timestamp):
            path_saveimg_base = path_saveimg_base + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        path_pictures = [
            os.path.join(path_groundtruth, path_saveimg_base),
            os.path.join(path_observation, path_saveimg_base),
            os.path.join(path_result, path_saveimg_base)
        ]
        save_imgs(pictures = pictures, path_pictures = path_pictures, format = '.png')

        # =====================================
        # Plot graphs if necessary
        # =====================================
        if(result_output):
            x = np.arange(0, max_iter, 1)
            plt.title('PSNR')
            plt.plot(x, psnr_evolution)
            # plt.gca().set_yscale('log')
            # plt.plot(x, c_evolution) 
            plt.xlabel('iteration')
            plt.ylabel('PSNR')
            plt.show()
        
        timestamp_commandline = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(timestamp_commandline + '  (' + str(index+1) + '/' + str(len(path_images)) + ') '
            + ('Poisson alpha:' + str(poisson_alpha) if poisson_noise else 'Gaussian sigma:' + str(gaussian_nl)) + '  '
            + 'PSNR:' + str(psnr[index].round(3)).ljust(6, '0')
            + '    SSIM:' + str(ssim[index].round(3)).ljust(6, '0')
            + '   ' + filename)


    # =====================================
    # Save all results
    # =====================================
    timestamp_commandline = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
#    params = {'architecture':architecture, 'gamma1': gamma1, 'gamma2': gamma2, 'alpha_n': alpha_n, 'gaussian_nl':gaussian_nl, 'sp_nl':sp_nl, 'poisson-noise':poisson_noise, 'poisson_alpha':poisson_alpha, 'alpha_n':alpha_n, 'alpha_s':alpha_s, 'max_iter':max_iter, 'myLambda': myLambda, 'r':r,  'deg_op': deg_op, 'method':method, 'ch':ch, 'm1':m1, 'm2':m2, 'gammaInADMMStep1':gammaInADMMStep1}
    algorithm, denoiser = get_algorithm_denoiser (method)
    summary = {'Average_PSNR':np.mean(psnr), 'PSNR':psnr, 'Average_SSIM':np.mean(ssim), 'SSIM' : ssim, 'Average_time':np.average(cpu_time) , 'Cpu_time': cpu_time, 'algorithm' : algorithm, 'denoiser' : denoiser, 'architecture' : architecture}
    datas = {'experimental_settings' : experimental_settings_all, 'method' : method_all, 'configs' : configs_all, 'results' : results, 'summary' : summary}
    np.save(path_result + '\data' , datas)

    print(timestamp_commandline + '  Average_PSNR:' + str(np.mean(psnr).round(3)) + '  Average_SSIM:' + str(np.mean(ssim).round(3)) + '    Algorithm:' +  method + '   Observation:' + deg_op + '   Gaussian noise level:' + str(gaussian_nl).ljust(5, '0'))

    return datas






### TCI-Reply-Letter Round2 Blur Kernels (poisson)


def main():
    experiment_data_list = []
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}"
    folder_root = os.path.join(config['path_result'], folder_name)
    os.makedirs(folder_root, exist_ok=True)

    filepath_summary = os.path.join(folder_root, 'SUMMARY(' + str(datetime.datetime.now().strftime("%Y%m%d %H%M%S %f")) + ').txt')
    touch_textfile (filepath_summary)

    noise_level_list = [1, 2, 10]
    alpha_list = [0.82, 0.86, 0.92, 0.96, 1]
    obs_list = ['blur', 'random_sampling']
    method_list_P = ['A-Proposed',  'A-PnPPDS-DnCNN-wo-constraint', 'A-PnPPDS-DnCNN-clipping-layer', 'A-PDS-TV', 'A-PnPPDS-unstable-DnCNN']
    method_list_G = ['A-PnPFBS-DnCNN', 'A-RED-DnCNN']
    method_list_P = []
    method_list_G = ['C-Proposed', 'C-PnPPDS-DnCNN-wo-constraint', 'C-PnPPDS-DnCNN-clipping-layer','C-PnP-unstable-DnCNN', 'C-RED-DnCNN']
    method_list_G = ['C-Proposed', 'C-PnPPDS-DnCNN-wo-constraint', 'C-PnPPDS-DnCNN-clipping-layer','C-PnP-unstable-DnCNN', 'C-RED-DnCNN', 'C-PnPADMM-DnCNN']
    myLambda_list = [[0.002, 0.002, 0.0015, 0.0015],
                     [0.00125, 0.00125, 0.001, 0.001]
                     ]
    myLambda_coef_list = [1, 1, 1, 4000, 0.5, 400]
    myLambda_coef_list = [1, 1, 1, 4000, 0.5, 400]
#    method_list_G = ['C-PnPADMM-DnCNN']
    lambADMM_list = [0.01, 0.01]
    myLambda_list_ADMM = [
                        [15, 15, 10, 10,],
                        [20, 20, 50, 50]
                     ]
    #method_list_G = ['C-PnPPDS-DnCNN-clipping-layer']
#    myLambda_coef_list = [1]
#    obs_option_list = [ ['blur_1', 'blur_2', 'blur_3', 'blur_4', 'blur_5', 'blur_6', 'blur_7', 'blur_8', 'gaussian_1_6', 'square_7'],[0.8], ]
    obs_option_list = [['blur_1'], [0.8]]
    architecture = 'DnCNN_nobn_nch_1_nlev_0.01_journal'

    for nl_ind, nl in enumerate(noise_level_list):
        for obs_ind, obs in enumerate(obs_list):
            for obs_option in obs_option_list[obs_list.index(obs)]:
                if (obs == 'blur'):
                    max_iter = 4800
                    r = 1
                    blur_kernel = obs_option
                elif (obs == 'random_sampling'):
                    max_iter = 12000
                    r = obs_option
                    blur_kernel = 'blur_1'
                settings =  {'gaussian_nl' : 0, 'sp_nl' : 0, 'poisson_noise' : True, 'poisson_alpha' : nl, 'deg_op' : obs, 'r' : r, 'blur_kernel' : blur_kernel}
                configs = {'add_timestamp' : False, 'ch' : 1}

                for method_G in method_list_G:
                    for coef in [0.5, 1, 1.5]:
                        if (method_G == 'C-PnP-unstable-DnCNN'):
                            architecture = 'dncnn_15'
                        elif (method_G.find('DRUNet') != -1):
                            architecture = 'drunet_color'
                        else:
                            architecture = 'DnCNN_nobn_nch_1_nlev_0.01_journal'
    #                    for myLambda in [0.001, 0.00125,0.0015, 0.002]:
                        if (method_G == 'C-PnPADMM-DnCNN'):
                            myLambda = myLambda_list_ADMM[obs_ind][nl_ind]
                        else:
                            myLambda = myLambda_list[obs_ind][nl_ind]
                            myLambda *= myLambda_coef_list[method_list_G.index(method_G)]
                        myLambda*=coef
                        m1 = 50
                        m2 = 5
                        gammaInADMMStep1 = lambADMM_list[obs_ind]
                        if (method_G == 'C-RED-DnCNN' or method_G == 'C-Proposed' or method_G == 'C-PnPPDS-DnCNN-wo-constraint' or method_G == 'C-PnPPDS-DnCNN-clipping-layer'):
                            gamma1 = 0.5
                            gamma2 = 1 / (gamma1 * 2)
                        else:
                            gamma1 = 0.0005
                            gamma2 = 1000
                        experiment_data = {'settings' : settings, 'method' : {'method' : method_G, 'max_iter' : max_iter, 'myLambda': myLambda, 'gamma1' :  gamma1, 'gamma2' :  gamma2, 'm1':m1, 'm2':m2, 'alpha_n' : 1, 'gammaInADMMStep1': gammaInADMMStep1,'architecture' : architecture}, 'configs' : configs}
                        param_str = f'lamb_{(myLambda):.3g}'

                        path_save = {}
                        path_save['result'] = get_result_folder_name (experiment_data, folder_root, param_str)
                        path_save['observation'] = get_observation_folder_name (experiment_data, folder_root)
                        path_save['groundtruth'] = get_groundtruth_folder_name (folder_root)
                        experiment_data['path_save'] = path_save
                        experiment_data_list.append (experiment_data)

 
    for experiment_data in experiment_data_list:
        data = test_all_images(experiment_data['settings'], experiment_data['method'], experiment_data['configs'], experiment_data['path_save'])
        if (data != None): write_textfile (filepath_summary, data)
    if (data != None): add_footer_textfile (filepath_summary, data)


if (__name__ == '__main__'):

    main()

