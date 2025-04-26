#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
from astro.metrics import db_eval_iou, db_eval_boundary

# python evaluate-gt-pred.py --gt_dir /UBC-O/joy0921/Desktop/Dataset/Evaluations/2D-masks/gt/SB230/550 --pred_dir /UBC-O/joy0921/Desktop/Dataset/Evaluations/2D-masks/unetr-outputs/SB230/550
def load_binary_mask(path):
    """
    Load a mask image and convert to binary boolean array.
    Any non-zero pixel is treated as foreground.
    """
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.uint8)
    return arr > 0

def evaluate(gt_dir, pred_dir, bound_th=0.008):
    # list all .png files in gt_dir, sorted by filename
    frames = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))
    if not frames:
        raise RuntimeError(f"No .png files found in ground truth directory: {gt_dir}")

    iou_list = []
    f_list   = []
    jf_list  = []

    for fname in frames:
        gt_path   = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            print(f"[WARNING] prediction missing for frame {fname}, skipping.")
            continue

        gt_mask   = load_binary_mask(gt_path)
        pred_mask = load_binary_mask(pred_path)

        # compute Jaccard IoU
        iou = db_eval_iou(gt_mask, pred_mask)
        # compute boundary F‐measure
        f   = db_eval_boundary(gt_mask, pred_mask, bound_th=bound_th)
        # compute combined J&F
        jf  = (iou + f) / 2.0

        iou_list.append(iou)
        f_list.append(f)
        jf_list.append(jf)

        print(f"Frame {fname:>7} │ IoU = {iou:.4f} │ F = {f:.4f} │ J&F = {jf:.4f}")

    if iou_list:
        mean_iou = np.mean(iou_list)
        mean_f   = np.mean(f_list)
        mean_jf  = np.mean(jf_list)

        print("\n=== Summary ===")
        print(f"Mean IoU : {mean_iou:.4f}")
        print(f"Mean F   : {mean_f:.4f}")
        print(f"Mean J&F : {mean_jf:.4f}")
    else:
        print("No frames were evaluated.")

def main():
    p = argparse.ArgumentParser(description="Evaluate segmentation masks using Jaccard IoU, boundary F-score, and combined J&F")
    p.add_argument('--gt_dir',   type=str, required=True, help="Directory of ground truth masks (0.png–255.png)")
    p.add_argument('--pred_dir', type=str, required=True, help="Directory of predicted masks (0.png–255.png)")
    p.add_argument('--bound_th', type=float, default=0.008, help="Boundary threshold (as fraction of image diagonal)")
    args = p.parse_args()

    evaluate(args.gt_dir, args.pred_dir, bound_th=args.bound_th)

if __name__ == "__main__":
    main()

