import os
import shutil

def copy_selected_files():
    src_folder = './result/result-TCI-reply-discussion1/Gaussian/'
    nlev_list = ['0.0075', '0.01', '0.05', '0.1', 'mixed0.1']
    setting_text = 'blur_0.005'
    img_file = 'ILSVRC2012_val_00002289.JPEG.png'
    file_list = [
        'GROUND_TRUTH_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_' + setting_text + '_300_(' + img_file + ')_alpha10000_lambda10000.png',
        'OBSERVATION_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_0.01_dict_' + setting_text + '_300_(' + img_file + ')_alpha10000_lambda10000.png',
    ]
    for nlev in nlev_list:
        file_list.append(
        'RESULT_A-Proposed_reply_DnCNN_nobn_nch_3_nlev_' + nlev + '_dict_' + setting_text + '_300_(' + img_file + ')_alpha10000_lambda10000.png'
        )
    """
    指定されたフォルダ `src_folder` 内の `file_list` に含まれるファイルのみを、
    `src_folder/pickup` にコピーする。

    Parameters:
        src_folder (str): 元のフォルダのパス
        file_list (list of str): コピーしたいファイル名（拡張子付き）

    Returns:
        None
    """

    # "pickup" フォルダのパス
    pickup_folder = os.path.join(src_folder, "pickup")
    os.makedirs(pickup_folder, exist_ok=True)

    for filename in file_list:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(pickup_folder, filename)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")
        else:
            print(f"Skipped (not found): {filename}")

if (__name__ == '__main__'):
    copy_selected_files()