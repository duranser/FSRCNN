from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fsrcnn.dataset import SRImageDataset
from fsrcnn.model import FSRCNNDeconv, count_parameters
from fsrcnn.utils import (
    Y_CONVENTION,
    assert_checkpoint_y_convention,
    bicubic_baseline_tensor,
    calc_psnr,
    calc_ssim,
    clamp_model_output,
    compose_matlab_y_with_bicubic_chroma,
    crop_to_scale,
    load_checkpoint,
    make_lr_rgb_from_hr,
    open_rgb,
    save_thesis_images,
    tensor_to_pil,
)


@torch.no_grad()
def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state, checkpoint = load_checkpoint(args.checkpoint, device)
    assert_checkpoint_y_convention(
        checkpoint, allow_legacy=args.allow_legacy_checkpoint
    )
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    scale = args.scale if args.scale is not None else int(config.get("scale", 2))
    d = args.d if args.d is not None else int(config.get("d", 56))
    s = args.s if args.s is not None else int(config.get("s", 12))
    m = args.m if args.m is not None else int(config.get("m", 4))
    channels = args.channels if args.channels is not None else int(config.get("channels", 1))
    deconv_kernel = (
        args.deconv_kernel
        if args.deconv_kernel is not None
        else int(config.get("deconv_kernel", 9))
    )
    deconv_std = float(config.get("deconv_std", 0.001))

    model = FSRCNNDeconv(scale, d, s, m, channels, deconv_kernel, deconv_std).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Device       : {device}")
    print(f"Model        : FSRCNN Deconv x{scale}, d={d}, s={s}, m={m}, channels={channels}")
    print(f"Parameters   : {count_parameters(model):,}")
    print(f"Y convention : {Y_CONVENTION if channels == 1 else 'RGB [0,1]'}")
    print(f"Border shave : {scale} pixels")

    all_sr_psnr, all_sr_ssim = [], []
    all_bicubic_psnr, all_bicubic_ssim = [], []

    for test_dir in args.test_dirs:
        dataset = SRImageDataset(test_dir, scale, channels)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        sr_psnrs, sr_ssims = [], []
        bicubic_psnrs, bicubic_ssims = [], []
        save_dir = Path(args.save_images) / Path(test_dir).name if args.save_images else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for lr, hr, path in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = clamp_model_output(model(lr), channels)
            if sr.shape != hr.shape:
                raise RuntimeError(
                    f"Test shape mismatch: SR={tuple(sr.shape)}, HR={tuple(hr.shape)}"
                )
            bicubic_tensor = clamp_model_output(
                bicubic_baseline_tensor(lr, hr.shape[-2:]), channels
            )
            sr_psnrs.append(calc_psnr(sr, hr, shave=scale))
            sr_ssims.append(calc_ssim(sr, hr, shave=scale))
            bicubic_psnrs.append(calc_psnr(bicubic_tensor, hr, shave=scale))
            bicubic_ssims.append(calc_ssim(bicubic_tensor, hr, shave=scale))

            if save_dir:
                source_path = Path(path[0])
                gt_rgb = crop_to_scale(open_rgb(source_path), scale)
                lr_rgb = make_lr_rgb_from_hr(gt_rgb, scale)
                sr_pil = tensor_to_pil(sr)
                if channels == 1:
                    sr_rgb = compose_matlab_y_with_bicubic_chroma(sr_pil, lr_rgb)
                else:
                    sr_rgb = sr_pil.convert("RGB")
                save_thesis_images(
                    gt_rgb, lr_rgb, sr_rgb, save_dir,
                    source_path.stem, scale, "FSRCNN_Deconv"
                )

        avg_sr_psnr = sum(sr_psnrs) / len(sr_psnrs)
        avg_sr_ssim = sum(sr_ssims) / len(sr_ssims)
        avg_bicubic_psnr = sum(bicubic_psnrs) / len(bicubic_psnrs)
        avg_bicubic_ssim = sum(bicubic_ssims) / len(bicubic_ssims)
        all_sr_psnr.append(avg_sr_psnr)
        all_sr_ssim.append(avg_sr_ssim)
        all_bicubic_psnr.append(avg_bicubic_psnr)
        all_bicubic_ssim.append(avg_bicubic_ssim)

        print(f"\n{Path(test_dir).name} ({len(sr_psnrs)} images)")
        print(f"  Bicubic : PSNR={avg_bicubic_psnr:.3f} dB, SSIM={avg_bicubic_ssim:.4f}")
        print(f"  Model   : PSNR={avg_sr_psnr:.3f} dB, SSIM={avg_sr_ssim:.4f}")

    if len(all_sr_psnr) > 1:
        print("\nUnweighted dataset mean")
        print(
            f"  Bicubic : PSNR={sum(all_bicubic_psnr)/len(all_bicubic_psnr):.3f} dB, "
            f"SSIM={sum(all_bicubic_ssim)/len(all_bicubic_ssim):.4f}"
        )
        print(
            f"  Model   : PSNR={sum(all_sr_psnr)/len(all_sr_psnr):.3f} dB, "
            f"SSIM={sum(all_sr_ssim)/len(all_sr_ssim):.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Test FSRCNN Deconv.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dirs", nargs="+", default=["data/Set5", "data/Set14"])
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None, choices=[1, 3])
    parser.add_argument("--deconv-kernel", type=int, default=None)
    parser.add_argument("--save-images", default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
