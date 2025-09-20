import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

# v7.3 (HDF5) 用
try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

PREFERRED_NAMES = ["kernel", "K", "psf"]

def load_mat_array(path):
    """
    .mat から 2D の数値配列を取得する。
    1) 'kernel','K','psf' を優先
    2) なければ最大サイズの2D数値配列を自動選択
    v7.3(HDF5)にも対応
    """
    # まずは v7.3 以外の標準 .mat を試す
    try:
        mdict = loadmat(path)
        candidates = {k: v for k, v in mdict.items() if not k.startswith("__")}
        for name in PREFERRED_NAMES:
            if name in candidates and isinstance(candidates[name], np.ndarray) and candidates[name].ndim == 2:
                return np.array(candidates[name], dtype=float), name
        best_name, best_arr, best_size = None, None, -1
        for k, v in candidates.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
                size = v.size
                if size > best_size:
                    best_name, best_arr, best_size = k, v, size
        if best_arr is not None:
            return np.array(best_arr, dtype=float), best_name
        raise ValueError("No 2D numeric array found in non-HDF5 .mat")
    except NotImplementedError:
        # おそらく v7.3(HDF5)
        if not HAS_H5PY:
            raise RuntimeError("This .mat seems to be v7.3 (HDF5). Install h5py to read it.")
        with h5py.File(path, "r") as f:
            for name in PREFERRED_NAMES:
                if name in f:
                    arr = np.array(f[name]).squeeze()
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        return arr.astype(float), name
            best_name, best_arr, best_size = None, None, -1
            def visit(name, obj):
                nonlocal best_name, best_arr, best_size
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = np.array(obj).squeeze()
                        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                            size = arr.size
                            if size > best_size:
                                best_name, best_arr, best_size = name, arr, size
                    except Exception:
                        pass
            f.visititems(lambda n, o: visit(n, o))
            if best_arr is not None:
                return best_arr.astype(float), best_name
            raise ValueError("No 2D numeric array found in HDF5 .mat")

def normalize_to_unit_interval(A):
    """最小=0(黒), 最大=1(白) になるように正規化。一定配列は0に。"""
    A = np.asarray(A, dtype=float)
    amin, amax = np.nanmin(A), np.nanmax(A)
    if not np.isfinite(amin) or not np.isfinite(amax):
        finite = A[np.isfinite(A)]
        if finite.size == 0:
            return np.zeros_like(A)
        amin, amax = finite.min(), finite.max()
    denom = (amax - amin)
    if denom <= 0:
        return np.zeros_like(A)
    return (A - amin) / denom

def save_kernel_image(Knorm, out_path_base, fmt, dpi=600, transparent=False, figsize=2.5):
    """
    正規化済みカーネルを画像保存。
    fmt: 'eps' | 'pdf' | 'png'
    """
    fig = plt.figure(figsize=(figsize, figsize), dpi=300 if fmt != "png" else dpi)
    ax = fig.add_subplot(111)
    ax.imshow(Knorm, cmap="gray", interpolation="nearest", aspect="equal")
    ax.axis("off")
    plt.tight_layout(pad=0)

    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    out_path = f"{out_path_base}.{fmt}"

    if fmt == "png":
        plt.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=transparent)
    elif fmt == "pdf":
        plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0)
    elif fmt == "eps":
        plt.savefig(out_path, format="eps", bbox_inches="tight", pad_inches=0)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    plt.close(fig)
    return out_path

def process_folder(in_dir, out_dir, formats, dpi, transparent, figsize):
    mats = glob.glob(os.path.join(in_dir, "**", "*.mat"), recursive=True)
    if not mats:
        print(f"[WARN] No .mat files found under: {in_dir}")
        return
    print(f"[INFO] Found {len(mats)} .mat files.")
    for i, mpath in enumerate(mats, 1):
        try:
            arr, vname = load_mat_array(mpath)
            Knorm = normalize_to_unit_interval(arr)
            rel = os.path.relpath(mpath, in_dir)
            base_noext = os.path.splitext(rel)[0]
            out_base = os.path.join(out_dir, base_noext)
            for fmt in formats:
                out_path = save_kernel_image(Knorm, out_base, fmt, dpi=dpi, transparent=transparent, figsize=figsize)
                print(f"[{i:03d}/{len(mats)}] OK  {mpath}  ->  {out_path}  (var: {vname})")
        except Exception as e:
            print(f"[{i:03d}/{len(mats)}] ERR {mpath}  ({e})", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Convert MATLAB .mat kernels to images (min=black, max=white).")
    parser.add_argument("in_dir", help="Input folder containing .mat files (searched recursively).")
    parser.add_argument("--out_dir", default=None, help="Output folder (default: <in_dir>/img_out)")
    parser.add_argument(
        "--formats",
        default="eps",
        help="Comma-separated output formats: eps,pdf,png (default: eps)"
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for PNG (ignored for EPS/PDF). Default: 600")
    parser.add_argument("--transparent", action="store_true", help="Make PNG background transparent.")
    parser.add_argument("--figsize", type=float, default=2.5, help="Figure size in inches (square). Default: 2.5")
    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(in_dir, "img_out")
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    for f in formats:
        if f not in {"eps", "pdf", "png"}:
            raise ValueError(f"Unsupported format in --formats: {f}")

    process_folder(in_dir, out_dir, formats, dpi=args.dpi, transparent=args.transparent, figsize=args.figsize)

if __name__ == "__main__":
    main()
