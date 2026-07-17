from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# MATLAB rgb2ycbcr uint8 luminance occupies code values 16..235.
MATLAB_Y_MIN_8BIT = 16
MATLAB_Y_MAX_8BIT = 235
MATLAB_Y_MIN = MATLAB_Y_MIN_8BIT / 255.0
MATLAB_Y_MAX = MATLAB_Y_MAX_8BIT / 255.0
Y_CONVENTION = "MATLAB rgb2ycbcr limited-range Y [16,235]/255"


def bicubic():
    try:
        return Image.Resampling.BICUBIC
    except AttributeError:
        return Image.BICUBIC


def nearest():
    try:
        return Image.Resampling.NEAREST
    except AttributeError:
        return Image.NEAREST


def flip_left_right():
    try:
        return Image.Transpose.FLIP_LEFT_RIGHT
    except AttributeError:
        return Image.FLIP_LEFT_RIGHT


def flip_top_bottom():
    try:
        return Image.Transpose.FLIP_TOP_BOTTOM
    except AttributeError:
        return Image.FLIP_TOP_BOTTOM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in IMG_EXTENSIONS:
        return [root]
    images = [path for path in root.rglob("*") if path.suffix.lower() in IMG_EXTENSIONS]
    images.sort()
    return images


def open_rgb(path) -> Image.Image:
    from PIL import ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def crop_to_scale(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    width -= width % scale
    height -= height % scale
    if width <= 0 or height <= 0:
        raise ValueError(f"Image {image.size} is too small for scale x{scale}")
    return image.crop((0, 0, width, height))


def rgb_to_matlab_y(image: Image.Image) -> Image.Image:
    """Convert RGB uint8 to MATLAB-compatible uint8 Y.

    This implements the limited-range BT.601 transform used by MATLAB's
    rgb2ycbcr for uint8 data:

        Y = 16 + 65.481 R + 128.553 G + 24.966 B

    where R, G and B are normalized to [0, 1]. The returned image is an
    8-bit single-channel PIL image whose nominal range is [16, 235].
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    y = (
        16.0
        + 65.481 * rgb[..., 0]
        + 128.553 * rgb[..., 1]
        + 24.966 * rgb[..., 2]
    )
    # MATLAB's uint8 path rounds and saturates to the uint8 range.
    y = np.clip(np.rint(y), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(y, mode="L")


def matlab_y_to_full_range_pil(image: Image.Image) -> Image.Image:
    """Map limited-range MATLAB Y [16,235] to full-range display Y [0,255]."""
    y = np.asarray(image.convert("L"), dtype=np.float64)
    full = (y - MATLAB_Y_MIN_8BIT) * (255.0 / 219.0)
    full = np.clip(np.rint(full), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(full, mode="L")


def to_train_channel(image: Image.Image, channels: int) -> Image.Image:
    if channels == 1:
        return rgb_to_matlab_y(image)
    if channels == 3:
        return image.convert("RGB")
    raise ValueError("channels must be 1 or 3")


def downsample_bicubic(hr_image: Image.Image, scale: int) -> Image.Image:
    """Create LR from the complete HR image using Pillow bicubic.

    Every stage in this project calls this same function, so training,
    validation and testing use identical degradation order and implementation.
    """
    width, height = hr_image.size
    if width % scale != 0 or height % scale != 0:
        raise ValueError("HR image must be cropped to a multiple of scale first")
    return hr_image.resize((width // scale, height // scale), bicubic())


def make_aligned_pair(
    hr_rgb: Image.Image,
    scale: int,
    channels: int,
) -> tuple[Image.Image, Image.Image]:
    """Return an aligned full-image LR/HR pair in one consistent color space."""
    hr_rgb = crop_to_scale(hr_rgb, scale)
    hr = to_train_channel(hr_rgb, channels)
    lr = downsample_bicubic(hr, scale)
    return lr, hr


def make_lr_rgb_from_hr(hr_rgb: Image.Image, scale: int) -> Image.Image:
    """Create the color LR visualization from the complete HR RGB image."""
    hr_rgb = crop_to_scale(hr_rgb, scale)
    return hr_rgb.resize((hr_rgb.width // scale, hr_rgb.height // scale), bicubic())


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = np.transpose(tensor.numpy(), (1, 2, 0))
    array = np.round(array * 255.0).astype(np.uint8)
    if array.shape[-1] == 1:
        return Image.fromarray(array[:, :, 0], mode="L")
    return Image.fromarray(array, mode="RGB")


def clamp_model_output(sr: torch.Tensor, channels: int) -> torch.Tensor:
    """Clamp model output to the valid range for the selected channel space."""
    if channels == 1:
        return sr.clamp(MATLAB_Y_MIN, MATLAB_Y_MAX)
    if channels == 3:
        return sr.clamp(0.0, 1.0)
    raise ValueError("channels must be 1 or 3")


def compose_matlab_y_with_bicubic_chroma(
    sr_y_matlab: Image.Image,
    lr_rgb: Image.Image,
) -> Image.Image:
    """Compose limited-range reconstructed Y with bicubic LR chroma.

    Pillow's YCbCr merge expects a full-range display Y. Therefore, the model's
    MATLAB-compatible Y is mapped back to full range only for RGB visualization.
    Training and metric computation remain in limited-range MATLAB Y.
    """
    sr_y_full = matlab_y_to_full_range_pil(sr_y_matlab)
    _, cb, cr = lr_rgb.convert("YCbCr").split()
    cb = cb.resize(sr_y_full.size, bicubic())
    cr = cr.resize(sr_y_full.size, bicubic())
    return Image.merge("YCbCr", [sr_y_full, cb, cr]).convert("RGB")


def bicubic_baseline_tensor(lr: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Upscale a batch-size-one tensor through the same Pillow bicubic path."""
    if lr.ndim != 4 or lr.shape[0] != 1:
        raise ValueError("bicubic_baseline_tensor expects a batch-size-one NCHW tensor")
    lr_pil = tensor_to_pil(lr)
    up_pil = lr_pil.resize((output_size[1], output_size[0]), bicubic())
    return pil_to_tensor(up_pil).unsqueeze(0).to(device=lr.device, dtype=lr.dtype)


def assert_checkpoint_y_convention(checkpoint: dict, allow_legacy: bool = False) -> None:
    """Prevent accidental use of checkpoints trained on Pillow full-range Y."""
    if not isinstance(checkpoint, dict) or not checkpoint:
        if allow_legacy:
            return
        raise ValueError(
            "Checkpoint has no configuration metadata. It may use the old full-range "
            "Pillow Y convention. Use --allow-legacy-checkpoint only when you are certain."
        )
    config = checkpoint.get("config", {})
    convention = config.get("y_convention")
    if convention != Y_CONVENTION and not allow_legacy:
        raise ValueError(
            "Checkpoint Y convention is incompatible with this project. "
            f"Expected: {Y_CONVENTION!r}; found: {convention!r}. "
            "Retrain from scratch for MATLAB-compatible Y, or pass "
            "--allow-legacy-checkpoint at your own risk."
        )


def _match_and_shave(
    sr: torch.Tensor,
    hr: torch.Tensor,
    shave: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    sr = sr.detach().float()
    hr = hr.detach().float()
    if sr.ndim == 3:
        sr = sr.unsqueeze(0)
    if hr.ndim == 3:
        hr = hr.unsqueeze(0)
    if sr.shape != hr.shape:
        raise ValueError(f"Metric shape mismatch: SR={tuple(sr.shape)}, HR={tuple(hr.shape)}")
    height, width = sr.shape[-2:]
    if shave > 0 and height > 2 * shave and width > 2 * shave:
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]
    return sr, hr


def calc_psnr(sr: torch.Tensor, hr: torch.Tensor, shave: int = 0) -> float:
    """PSNR with peak value 1.0, matching uint8 peak 255 after normalization."""
    sr, hr = _match_and_shave(sr, hr, shave)
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def calc_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    shave: int = 0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    sr, hr = _match_and_shave(sr, hr, shave)
    _, channels, height, width = sr.shape
    if height < 2 or width < 2:
        return float("nan")
    actual_window = min(window_size, height, width)
    if actual_window % 2 == 0:
        actual_window -= 1
    actual_window = max(actual_window, 1)
    coordinates = torch.arange(
        actual_window,
        device=sr.device,
        dtype=sr.dtype,
    ) - actual_window // 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).view(1, 1, actual_window, actual_window)
    window = window.repeat(channels, 1, 1, 1)

    # Valid convolution avoids artificial zero-padding at image boundaries.
    mu1 = F.conv2d(sr, window, padding=0, groups=channels)
    mu2 = F.conv2d(hr, window, padding=0, groups=channels)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1 = F.conv2d(sr * sr, window, padding=0, groups=channels) - mu1_sq
    sigma2 = F.conv2d(hr * hr, window, padding=0, groups=channels) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, padding=0, groups=channels) - mu12
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return float(score.mean().item())


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch: int = 0,
    config=None,
    extra=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": config or {},
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, {}


def append_metrics_csv(csv_path, row: Dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = list(row.keys())
    if exists:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            header = next(reader, None)
        if header:
            fieldnames = header
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})

def require_scale_divisible(image: Image.Image, scale: int, image_name: str = "image") -> None:
    """Require exact GT -> LR -> SR geometry without cropping the ground truth."""
    width, height = image.size
    if width % scale != 0 or height % scale != 0:
        raise ValueError(
            f"{image_name} size {image.size} is not divisible by scale x{scale}. "
            "Inference preserves the ground-truth dimensions, so width and height "
            "must both be divisible by the scale factor."
        )


def make_gt_inference_pair(
    gt_rgb: Image.Image,
    scale: int,
    channels: int,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """Create exact-size GT, LR visualization, model LR, and model GT images."""
    gt_rgb = gt_rgb.convert("RGB")
    require_scale_divisible(gt_rgb, scale, "Ground-truth image")
    gt_model = to_train_channel(gt_rgb, channels)
    lr_model = downsample_bicubic(gt_model, scale)
    lr_rgb = gt_rgb.resize(
        (gt_rgb.width // scale, gt_rgb.height // scale),
        bicubic(),
    )
    return gt_rgb, lr_rgb, lr_model, gt_model


def save_thesis_images(
    gt_rgb: Image.Image,
    lr_rgb: Image.Image,
    sr_rgb: Image.Image,
    output_dir,
    stem: str,
    scale: int,
    model_label: str,
) -> dict[str, Path]:
    """Save LR, nearest-upscaled LR, original GT, and SR at GT size."""
    expected_lr = (gt_rgb.width // scale, gt_rgb.height // scale)
    if lr_rgb.size != expected_lr:
        raise ValueError(
            f"LR size {lr_rgb.size} does not match expected {expected_lr}"
        )
    if sr_rgb.size != gt_rgb.size:
        raise ValueError(
            f"SR size {sr_rgb.size} does not match GT size {gt_rgb.size}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nearest_rgb = lr_rgb.resize(gt_rgb.size, nearest())
    prefix = output_dir / stem
    paths = {
        "lr": Path(f"{prefix}_LR_x{scale}.png"),
        "nearest": Path(f"{prefix}_LR_Nearest_x{scale}.png"),
        "gt": Path(f"{prefix}_GT.png"),
        "sr": Path(f"{prefix}_{model_label}_x{scale}.png"),
    }
    lr_rgb.save(paths["lr"])
    nearest_rgb.save(paths["nearest"])
    gt_rgb.save(paths["gt"])
    sr_rgb.save(paths["sr"])
    return paths


def make_project_dirs(base_dir=".") -> list[Path]:
    """Create the standard data, run, and result directories."""
    base = Path(base_dir)
    folders = [
        "data/91-image",
        "data/Validation",
        "data/Set5",
        "data/Set14",
        "runs",
        "results",
    ]
    created = []
    for relative in folders:
        folder = base / relative
        folder.mkdir(parents=True, exist_ok=True)
        created.append(folder)
        print(f"created {folder}")
    print("\nPut training HR images in data/91-image.")
    print("A separate validation set may be placed in data/Validation.")
    print("Put benchmark HR images in data/Set5 and data/Set14.")
    return created


def check_y_pipeline() -> None:
    """Verify the MATLAB-compatible limited-range Y conversion."""
    expected = {
        "black": ((0, 0, 0), 16),
        "white": ((255, 255, 255), 235),
        "red": ((255, 0, 0), 81),
        "green": ((0, 255, 0), 145),
        "blue": ((0, 0, 255), 41),
    }
    print("MATLAB-compatible Y conversion check")
    all_ok = True
    for name, (rgb, target) in expected.items():
        image = Image.new("RGB", (1, 1), rgb)
        value = int(np.asarray(rgb_to_matlab_y(image))[0, 0])
        ok = value == target
        all_ok = all_ok and ok
        print(
            f"  {name:5s} RGB={rgb} -> Y={value:3d} | "
            f"expected={target:3d} | {'OK' if ok else 'FAIL'}"
        )
    print(f"Nominal Y range: [{MATLAB_Y_MIN_8BIT}, {MATLAB_Y_MAX_8BIT}]")
    if not all_ok:
        raise RuntimeError("MATLAB-compatible Y conversion self-check failed")
    print("Y conversion self-check passed.")


def inspect_epoch_checkpoints(epoch_dir) -> None:
    """Inspect the actual train.py checkpoint structure."""
    epoch_dir = Path(epoch_dir)
    checkpoint_paths = sorted(epoch_dir.glob("epoch_*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No epoch_*.pth checkpoints found in {epoch_dir}"
        )

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = checkpoint.get("epoch", "unknown")
        config = checkpoint.get("config", {})
        extra = checkpoint.get("extra", {})
        metrics = extra.get("metrics", {})

        print(f"\n{checkpoint_path.name}")
        print(f"Epoch: {epoch}")
        if config.get("model"):
            print(f"  model: {config['model']}")
        if config.get("y_convention"):
            print(f"  y_convention: {config['y_convention']}")

        train_loss = extra.get("train_loss")
        best_psnr = extra.get("best_psnr")
        best_ssim = extra.get("best_ssim")
        if train_loss is not None:
            print(f"  train_loss: {float(train_loss):.6f}")
        if best_psnr is not None:
            print(f"  best_val_psnr: {float(best_psnr):.3f} dB")
        if best_ssim is not None:
            print(f"  best_val_ssim: {float(best_ssim):.4f}")

        for dataset_name, values in metrics.items():
            parts = []
            if values.get("psnr") is not None:
                parts.append(f"PSNR={float(values['psnr']):.3f} dB")
            if values.get("ssim") is not None:
                parts.append(f"SSIM={float(values['ssim']):.4f}")
            if values.get("num_images") is not None:
                parts.append(f"images={values['num_images']}")
            print(f"  {dataset_name}: {' '.join(parts)}")


def _utils_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FSRCNN utility commands.")
    commands = parser.add_subparsers(dest="command", required=True)

    make_parser = commands.add_parser("make-dirs")
    make_parser.add_argument("--base-dir", default=".")

    commands.add_parser("check-y")

    inspect_parser = commands.add_parser("inspect-epochs")
    inspect_parser.add_argument("--epoch-dir", required=True)

    args = parser.parse_args()
    if args.command == "make-dirs":
        make_project_dirs(args.base_dir)
    elif args.command == "check-y":
        check_y_pipeline()
    elif args.command == "inspect-epochs":
        inspect_epoch_checkpoints(args.epoch_dir)


if __name__ == "__main__":
    _utils_cli()
