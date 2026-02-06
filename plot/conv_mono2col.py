import os
from PIL import Image

def main():
    # 入力フォルダと出力フォルダを指定
    input_folder = "./result/TCI-round2-poisson-2025-09-29_21-32-56/report/"   # モノクロ画像が入っているフォルダ
    output_folder = "./result/TCI-round2-poisson-2025-09-29_21-32-56/report/col/" # 保存先フォルダ

    # 出力フォルダを作成（存在しない場合）
    os.makedirs(output_folder, exist_ok=True)

    # 入力フォルダ内のファイルを処理
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            # 画像を開く
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")  # モノクロ→カラー(RGB)

            # 出力先のパス
            save_path = os.path.join(output_folder, filename)

            # 保存
            img.save(save_path, "PNG")
            print("保存しました:", save_path)

    print("変換が完了しました。")


def convert_png_to_pdf():
    # 入力フォルダと出力フォルダを指定
    input_folder = "./result/2025-10-02_21-49-29/random_sampling_gaussian0.01/methods/A-Proposed_[DnCNN_nobn_nch_3_nlev_0.01_journal/alpha_1/"    # PNG画像が入っているフォルダ
    output_folder = "./result/2025-10-02_21-49-29/random_sampling_gaussian0.01/methods/A-Proposed_[DnCNN_nobn_nch_3_nlev_0.01_journal/alpha_1/pdf/" # 保存先フォルダ

    # 出力フォルダを作成（存在しない場合）
    os.makedirs(output_folder, exist_ok=True)

    # 入力フォルダ内のファイルを処理
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            # 画像を開く
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")  # PDFはRGBが推奨

            # 拡張子を.pdfに変換
            pdf_filename = os.path.splitext(filename)[0] + ".pdf"
            save_path = os.path.join(output_folder, pdf_filename)

            # PDFとして保存
            img.save(save_path, "PDF")

    print("PNG → PDF の変換が完了しました。")



if (__name__ == '__main__'):

    convert_png_to_pdf()

