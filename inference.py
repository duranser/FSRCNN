from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from fsrcnn.model import FSRCNNDeconv
from fsrcnn.utils import (
    Y_CONVENTION,
    assert_checkpoint_y_convention,
    calc_psnr,
    calc_ssim,
    clamp_model_output,
    compose_matlab_y_with_bicubic_chroma,
    list_images,
    load_checkpoint,
    make_gt_inference_pair,
    open_rgb,
    pil_to_tensor,
    save_thesis_images,
    tensor_to_pil,
)


@torch.no_grad()
def infer_one(model, gt_path, output_dir, device, channels, scale):
    model.eval()
    gt_rgb, lr_rgb, lr_model, gt_model = make_gt_inference_pair(
        open_rgb(gt_path), scale, channels
    )
    lr_tensor = pil_to_tensor(lr_model).unsqueeze(0).to(device)
    gt_tensor = pil_to_tensor(gt_model).unsqueeze(0).to(device)
    sr_tensor = clamp_model_output(model(lr_tensor), channels)

    if sr_tensor.shape != gt_tensor.shape:
        raise RuntimeError(
            f"Inference shape mismatch: SR={tuple(sr_tensor.shape)}, "
            f"GT={tuple(gt_tensor.shape)}"
        )

    if channels == 1:
        sr_rgb = compose_matlab_y_with_bicubic_chroma(
            tensor_to_pil(sr_tensor), lr_rgb
        )
    else:
        sr_rgb = tensor_to_pil(sr_tensor).convert("RGB")

    paths = save_thesis_images(
        gt_rgb, lr_rgb, sr_rgb, output_dir, Path(gt_path).stem,
        scale, "FSRCNN_Deconv"
    )
    metrics = {
        "psnr": calc_psnr(sr_tensor, gt_tensor, shave=scale),
        "ssim": calc_ssim(sr_tensor, gt_tensor, shave=scale),
    }
    return paths, metrics, gt_rgb.size, lr_rgb.size


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

    print(f"Device       : {device}")
    print(f"Model        : FSRCNN Deconv x{scale}, d={d}, s={s}, m={m}, channels={channels}")
    print(f"Y convention : {Y_CONVENTION if channels == 1 else 'RGB [0,1]'}")
    print("Input mode   : GT -> internally degraded LR -> SR at GT size")

    input_path = Path(args.input)
    output_root = Path(args.output)
    records = []

    if input_path.is_dir():
        images = list_images(input_path)
        if not images:
            raise FileNotFoundError(f"No images found in {input_path}")
        for gt_path in tqdm(images, desc="Inference from GT"):
            relative_parent = gt_path.relative_to(input_path).parent
            _, metrics, _, _ = infer_one(
                model, gt_path, output_root / relative_parent,
                device, channels, scale
            )
            records.append(metrics)
        print(f"Saved outputs under: {output_root}")
        print(
            f"Mean: PSNR={sum(x['psnr'] for x in records)/len(records):.3f} dB, "
            f"SSIM={sum(x['ssim'] for x in records)/len(records):.4f}"
        )
    else:
        paths, metrics, gt_size, lr_size = infer_one(
            model, input_path, output_root, device, channels, scale
        )
        print(f"GT size      : {gt_size[0]}x{gt_size[1]}")
        print(f"LR size      : {lr_size[0]}x{lr_size[1]}")
        print(f"SR size      : {gt_size[0]}x{gt_size[1]} (same as GT)")
        print(f"Metrics      : PSNR={metrics['psnr']:.3f} dB, SSIM={metrics['ssim']:.4f}")
        for name, path in paths.items():
            print(f"{name:8s}: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FSRCNN Deconv from a ground-truth image. The LR input is "
            "created internally by bicubic downsampling."
        )
    )
    parser.add_argument("--input", required=True, help="GT image or GT directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None, choices=[1, 3])
    parser.add_argument("--deconv-kernel", type=int, default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
