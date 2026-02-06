import os,sys

def get_csv_header ():
    myStr = ''
    myStr += 'Observation,'
    myStr += 'Gaussian_noise,'
    myStr += 'Poisson_alpha,'
    myStr += 'method,'
    myStr += 'algorithm,'
    myStr += 'denoiser(architecture),'
    myStr += 'PSNR,'
    myStr += 'SSIM,'
    myStr += 'AverageTime,'
    myStr += 'gamma1,'
    myStr += 'gamma2,'
    myStr += 'alpha_n,'
    myStr += 'myLambda,'
    myStr += 'max_iter,'
    myStr += 'm1,'
    myStr += 'm2,'
    myStr += 'r,'
    myStr += 'blur_kernel,'
    myStr += 'ch,'
    myStr += 'Result PSNR - Result SSIM - Observed PSNR - Observed SSIM (for each images) - CPU Time\n'
    return myStr

def get_csv_data (data):
    myStr = ''
    myStr += str(data['experimental_settings']['deg_op']) + ','
    myStr += str(data['experimental_settings']['gaussian_nl']) + ','
    myStr += str(data['experimental_settings']['poisson_alpha']) + ','
    myStr += str(data['method']['method']) + ','
    myStr += str(data['summary']['algorithm']) + ','
    myStr += str(data['summary']['denoiser']) + '(' + str(data['summary']['architecture']) + '),'
    myStr += str(data['summary']['Average_PSNR']) + ','
    myStr += str(data['summary']['Average_SSIM']) + ','
    myStr += str(data['summary']['Average_time']) + ','
    myStr += str(data['method']['gamma1']) + ','
    myStr += str(data['method']['gamma2']) + ','
    myStr += str(data['method']['alpha_n']) + ','
    myStr += str(data['method']['myLambda']) + ','
    myStr += str(data['method']['max_iter']) + ','
    myStr += str(data['method']['m1']) + ','
    myStr += str(data['method']['m2']) + ','
    myStr += str(data['experimental_settings']['r']) + ','
    myStr += str(data['experimental_settings']['blur_kernel']) + ','
    myStr += str(data['configs']['ch']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['PSNR']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['SSIM']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['PSNR_observation']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['SSIM_observation']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['CPU_time']) + ','
    return myStr

def get_csv_footer (data):
    myStr = ''
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['filename']) + ','
    return myStr

def touch_textfile (filepath):
    f = open(filepath, 'w')
    f.write(get_csv_header())
    f.close()
    return

def write_textfile (filepath, data):
    f = open(filepath, 'a')
    f.write(get_csv_data(data) + '\n')
    f.close()
    return

def add_footer_textfile (filepath, data):
    f = open(filepath, 'a')
    f.write(get_csv_footer(data) + '\n')
    f.close()
    return


def get_folder_name_setting(experiment_data, root):
    # 各カテゴリを取得
    settings = experiment_data.get('settings', {})

    # オペレータとノイズ情報
    deg_op = settings.get('deg_op', 'unknown_op')
    gaussian_nl = settings.get('gaussian_nl', 0)
    poisson_alpha = settings.get('poisson_alpha', 0)
    sp_nl = settings.get('sp_nl', 0)
    poisson_noise = settings.get('poisson_noise', False)

    # ノイズ文字列の構築
    if poisson_noise:
        noise_str = f"poisson{poisson_alpha}"
    elif gaussian_nl > 0:
        noise_str = f"gaussian{gaussian_nl}"
    elif sp_nl > 0:
        noise_str = f"sp{sp_nl}"
    else:
        noise_str = "clean"
    kernel = ''
    if deg_op == 'blur':
        kernel = f"_{settings.get('blur_kernel', 'unknown_kernel')}"
    r = ''
    if deg_op == 'random_sampling':
        r = f"_{settings.get('r', 'unknown_r')}"

    # パス構築
    folder_path = os.path.join(
        root,
        f"{deg_op}{r}{kernel}_{noise_str}"
    )

    return folder_path

def get_result_folder_name(experiment_data, root, param_str):
    base = get_folder_name_setting(experiment_data, root)
    method_arg = experiment_data.get('method', {})

    # 手法とアーキテクチャ
    method = method_arg.get('method', 'unknown_method')
    architecture = method_arg.get('architecture', 'unknown_arch')

    # パス構築
    folder_path = os.path.join(
        base,
        "methods",
        f"{method}_[{architecture}",
        param_str
    )

    # フォルダを作成（再帰的）
    if os.path.exists(folder_path):
        print(f"Warning: The folder '{folder_path}' already exists. The program will terminate.")
        sys.exit(1) 
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def get_observation_folder_name(experiment_data, root):
    base = get_folder_name_setting(experiment_data, root)

    # パス構築
    folder_path = os.path.join(
        base,
        "observation",
    )

    # フォルダを作成（再帰的）
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def get_groundtruth_folder_name(root):
    # パス構築
    folder_path = os.path.join(
        root,
        "groundtruth"
    )

    # フォルダを作成（再帰的）
    os.makedirs(folder_path, exist_ok=True)

    return folder_path