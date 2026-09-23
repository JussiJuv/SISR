import csv
import re
import argparse
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torchmetrics.image.fid import FrechetInceptionDistance

# --- NEW METRIC IMPORTS ---
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import DeepImageStructureAndTextureSimilarity
from sklearn.metrics import normalized_mutual_info_score
# --------------------------

# Match folders like checkpoint-49_val, checkpoint-99_val, ..., checkpoint-499_val
CHECKPOINT_DIR_PATTERNS = [
    re.compile(r"checkpoint-(\d+).*_val$"),
    re.compile(r"(\d+)_EMA_val$"),
]
PRED_FILENAME_PATTERN = re.compile(r"(\d+)_pred\.png$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=False, help="Optional single checkpoint folder to process")
    return parser.parse_args()


def get_metrics(lpips_model, dists_model, msssim_model, sr_tensor, gt_tensor):
    """Return LPIPS, PSNR, SSIM, DISTS, MS-SSIM, and NMI for a single SR/GT image pair."""
    # LPIPS expects images in [-1, 1]
    sr_lpips = sr_tensor.unsqueeze(0) * 2.0 - 1.0
    gt_lpips = gt_tensor.unsqueeze(0) * 2.0 - 1.0

    # Torchmetrics models expect [B, C, H, W] in [0, 1]
    sr_tm = sr_tensor.unsqueeze(0)
    gt_tm = gt_tensor.unsqueeze(0)

    with torch.no_grad():
        lpips_val = lpips_model(sr_lpips, gt_lpips).item()
        dists_val = dists_model(sr_tm, gt_tm).item()
        # MS-SSIM can fail if image is too small for 5 scales
        try:
            msssim_val = msssim_model(sr_tm, gt_tm).item()
        except Exception:
            msssim_val = 0.0

    sr_np = (sr_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    psnr_val = sk_psnr(gt_np, sr_np, data_range=255)
    ssim_val = sk_ssim(gt_np, sr_np, channel_axis=2, data_range=255, win_size=7)
    
    # NMI calculation
    nmi_val = normalized_mutual_info_score(gt_np.ravel(), sr_np.ravel())

    return lpips_val, psnr_val, ssim_val, dists_val, msssim_val, nmi_val


def update_fid(fid_metric, sr_tensor, gt_tensor):
    """Update FID with a single GT/SR image pair."""
    sr_u8 = (sr_tensor.clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0)
    gt_u8 = (gt_tensor.clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0)

    fid_metric.update(gt_u8, real=True)
    fid_metric.update(sr_u8, real=False)


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load an RGB image as a float tensor in [0, 1]."""
    return TF.to_tensor(Image.open(path).convert("RGB")).to(device)


def extract_checkpoint_id(name: str):
    for pattern in CHECKPOINT_DIR_PATTERNS:
        match = pattern.match(name)
        if match:
            return int(match.group(1))
    return None


def process_checkpoint(
    checkpoint_dir: Path,
    gt_dir: Path,
    lpips_model,
    dists_model,
    msssim_model,
    device: torch.device,
):
    """
    Process one checkpoint folder and return:
      - per-image rows
      - summary dict with average metrics and FID
    """
    checkpoint_id = f"{extract_checkpoint_id(checkpoint_dir.name):03d}"
    samples_dir = checkpoint_dir / "validation_samples"

    if not samples_dir.exists():
        print(f"WARNING: Missing validation_samples folder in {checkpoint_dir.name}", flush=True)
        return None

    model_images = sorted(samples_dir.glob("*.png"))
    tasks = []

    for pred_path in model_images:
        match = PRED_FILENAME_PATTERN.match(pred_path.name)
        if not match:
            continue

        img_id = match.group(1)
        gt_path = gt_dir / f"{img_id}.png"

        if not gt_path.exists():
            print(f"WARNING: GT not found for {checkpoint_dir.name}/{pred_path.name}", flush=True)
            continue

        tasks.append((pred_path, gt_path, img_id))

    if len(tasks) == 0:
        print(f"WARNING: No valid image pairs found for {checkpoint_dir.name}", flush=True)
        return None

    print(f"\n=== Processing {checkpoint_dir.name} ({len(tasks)} image pairs) ===", flush=True)

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    results = []

    # Write CSV immediately so the file appears as soon as the first image is processed
    csv_path = OUT_DIR / f"metrics_checkpoint_{checkpoint_id}.csv"
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Image_ID", "LPIPS", "PSNR", "SSIM", "DISTS", "MS-SSIM", "NMI"])
        f_csv.flush()

        for i, (pred_path, gt_path, img_id) in enumerate(tasks):
            sr = load_rgb_tensor(pred_path, device)
            gt = load_rgb_tensor(gt_path, device)

            if sr.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch: SR {sr.shape} vs GT {gt.shape} for image {img_id} "
                    f"in checkpoint {checkpoint_dir.name}"
                )

            lp, ps, ss, ds, ms, nm = get_metrics(lpips_model, dists_model, msssim_model, sr, gt)
            update_fid(fid_metric, sr, gt)

            row = [img_id, lp, ps, ss, ds, ms, nm]
            results.append(row)

            # append current row immediately
            writer.writerow(row)
            f_csv.flush()

            print(
                f"[{i + 1}/{len(tasks)}] {img_id} | "
                f"LPIPS: {lp:.4f}, PSNR: {ps:.2f}, SSIM: {ss:.4f}, "
                f"DISTS: {ds:.4f}, MS-SSIM: {ms:.4f}, NMI: {nm:.4f}",
                flush=True
            )

    results.sort(key=lambda x: int(x[0]))

    print(f"Saved per-image results to {csv_path}", flush=True)

    lpips_vals  = [(r[0], r[1]) for r in results]
    psnr_vals   = [(r[0], r[2]) for r in results]
    ssim_vals   = [(r[0], r[3]) for r in results]
    dists_vals  = [(r[0], r[4]) for r in results]
    msssim_vals = [(r[0], r[5]) for r in results]
    nmi_vals    = [(r[0], r[6]) for r in results]

    def get_stats(vals, higher_is_better):
        avg = float(np.mean([v[1] for v in vals]))
        if higher_is_better:
            worst = min(vals, key=lambda x: x[1])
            best = max(vals, key=lambda x: x[1])
        else:
            best = min(vals, key=lambda x: x[1])
            worst = max(vals, key=lambda x: x[1])
        return avg, best, worst

    lp_avg, lp_b, lp_w = get_stats(lpips_vals, False)
    ps_avg, ps_b, ps_w = get_stats(psnr_vals, True)
    ss_avg, ss_b, ss_w = get_stats(ssim_vals, True)
    ds_avg, ds_b, ds_w = get_stats(dists_vals, False)
    ms_avg, ms_b, ms_w = get_stats(msssim_vals, True)
    nm_avg, nm_b, nm_w = get_stats(nmi_vals, True)

    fid_val = float(fid_metric.compute().item())

    summary_lines = []
    summary_lines.append(f"===== SUMMARY: checkpoint-{checkpoint_id}_val =====\n")

    summary_lines.append("LPIPS:")
    summary_lines.append(f"   Avg : {lp_avg:.4f}")
    summary_lines.append(f"   Best (lowest) : {lp_b[1]:.4f} (Image {lp_b[0]})")
    summary_lines.append(f"   Worst (highest): {lp_w[1]:.4f} (Image {lp_w[0]})\n")

    summary_lines.append("PSNR:")
    summary_lines.append(f"   Avg : {ps_avg:.2f}")
    summary_lines.append(f"   Worst (lowest) : {ps_w[1]:.2f} (Image {ps_w[0]})")
    summary_lines.append(f"   Best (highest): {ps_b[1]:.2f} (Image {ps_b[0]})\n")

    summary_lines.append("SSIM:")
    summary_lines.append(f"   Avg : {ss_avg:.4f}")
    summary_lines.append(f"   Worst (lowest) : {ss_w[1]:.4f} (Image {ss_w[0]})")
    summary_lines.append(f"   Best (highest): {ss_b[1]:.4f} (Image {ss_b[0]})\n")

    summary_lines.append("DISTS:")
    summary_lines.append(f"   Avg : {ds_avg:.4f}")
    summary_lines.append(f"   Best (lowest) : {ds_b[1]:.4f} (Image {ds_b[0]})")
    summary_lines.append(f"   Worst (highest): {ds_w[1]:.4f} (Image {ds_w[0]})\n")

    summary_lines.append("MS-SSIM:")
    summary_lines.append(f"   Avg : {ms_avg:.4f}")
    summary_lines.append(f"   Worst (lowest) : {ms_w[1]:.4f} (Image {ms_w[0]})")
    summary_lines.append(f"   Best (highest): {ms_b[1]:.4f} (Image {ms_b[0]})\n")

    summary_lines.append("NMI:")
    summary_lines.append(f"   Avg : {nm_avg:.4f}")
    summary_lines.append(f"   Worst (lowest) : {nm_w[1]:.4f} (Image {nm_w[0]})")
    summary_lines.append(f"   Best (highest): {nm_b[1]:.4f} (Image {nm_b[0]})\n")

    summary_lines.append("FID:")
    summary_lines.append(f"   Overall : {fid_val:.4f}")

    txt_path = OUT_DIR / f"metrics_summary_checkpoint_{checkpoint_id}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved summary to {txt_path}", flush=True)

    return {
        "checkpoint": checkpoint_id,
        "lpips_avg": lp_avg,
        "psnr_avg": ps_avg,
        "ssim_avg": ss_avg,
        "dists_avg": ds_avg,
        "msssim_avg": ms_avg,
        "nmi_avg": nm_avg,
        "fid": fid_val,
        "csv_path": str(csv_path),
        "txt_path": str(txt_path),
    }


def main():
    global BASE_DIR, GT_DIR, OUT_DIR

    args = parse_args()
    BASE_DIR = Path(args.base_dir)
    GT_DIR = Path(args.gt_dir)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    else:
        OUT_DIR = BASE_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Loading Models (LPIPS, DISTS, MS-SSIM)...", flush=True)
    lpips_model = lpips.LPIPS(net="alex").to(device)
    # Instantiate new models on device
    dists_model = DeepImageStructureAndTextureSimilarity().to(device)
    msssim_model = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    checkpoint_dirs = []
    for path in sorted(BASE_DIR.iterdir()):
        if not path.is_dir():
            continue

        ckpt_id = extract_checkpoint_id(path.name)
        if ckpt_id is None:
            continue

        if args.checkpoint and path.name != args.checkpoint:
            continue
        checkpoint_dirs.append((ckpt_id, path))

    if len(checkpoint_dirs) == 0:
        print(f"No checkpoint folders matching 'checkpoint-*_val' found in {BASE_DIR}")
        return

    print(f"Found {len(checkpoint_dirs)} checkpoint folders.", flush=True)

    total_rows = []

    for _, checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: x[0]):
        summary = process_checkpoint(
            checkpoint_dir=checkpoint_dir,
            gt_dir=GT_DIR,
            lpips_model=lpips_model,
            dists_model=dists_model,
            msssim_model=msssim_model,
            device=device,
        )
        if summary is not None:
            total_rows.append(summary)

    if len(total_rows) == 0:
        print("No checkpoints produced valid metrics.")
        return

    total_rows.sort(key=lambda x: int(x["checkpoint"]))

    total_csv_path = OUT_DIR / "total_summary.csv"
    with open(total_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Checkpoint", "LPIPS", "PSNR", "SSIM", "DISTS", "MS-SSIM", "NMI", "FID"])
        for row in total_rows:
            writer.writerow([
                row["checkpoint"],
                f"{row['lpips_avg']:.6f}",
                f"{row['psnr_avg']:.6f}",
                f"{row['ssim_avg']:.6f}",
                f"{row['dists_avg']:.6f}",
                f"{row['msssim_avg']:.6f}",
                f"{row['nmi_avg']:.6f}",
                f"{row['fid']:.6f}",
            ])

    print(f"\nSaved aggregate summary to {total_csv_path}", flush=True)


if __name__ == "__main__":
    main()