from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsrcnn.dataset import SRImageDataset, SRPatchDataset
from fsrcnn.model import FSRCNNDeconv, count_parameters
from fsrcnn.utils import (
    Y_CONVENTION,
    append_metrics_csv,
    assert_checkpoint_y_convention,
    calc_psnr,
    calc_ssim,
    clamp_model_output,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


@torch.no_grad()
def evaluate(model, val_dirs, scale: int, channels: int, device):
    model.eval()
    metrics = {}
    for val_dir in val_dirs:
        dataset = SRImageDataset(val_dir, scale=scale, channels=channels)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        psnrs = []
        ssims = []
        for lr, hr, _ in loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = clamp_model_output(model(lr), channels)
            if sr.shape != hr.shape:
                raise RuntimeError(
                    f"Validation shape mismatch: SR={tuple(sr.shape)}, HR={tuple(hr.shape)}"
                )
            psnrs.append(calc_psnr(sr, hr, shave=scale))
            ssims.append(calc_ssim(sr, hr, shave=scale))
        metrics[Path(val_dir).name] = {
            "psnr": sum(psnrs) / len(psnrs),
            "ssim": sum(ssims) / len(ssims),
            "num_images": len(psnrs),
        }
    return metrics


def mean_metric(metrics, key):
    return sum(value[key] for value in metrics.values()) / max(len(metrics), 1)


def format_metrics(metrics):
    return " | ".join(
        f"{name}: PSNR={value['psnr']:.3f} dB, SSIM={value['ssim']:.4f}"
        for name, value in metrics.items()
    )


def csv_row(epoch, loss, metrics, best_psnr, best_ssim):
    row = {
        "epoch": epoch,
        "train_loss": loss,
        "mean_val_psnr": mean_metric(metrics, "psnr") if metrics else "",
        "mean_val_ssim": mean_metric(metrics, "ssim") if metrics else "",
        "best_val_psnr": best_psnr,
        "best_val_ssim": best_ssim,
    }
    for name, value in metrics.items():
        row[f"{name}_psnr"] = value["psnr"]
        row[f"{name}_ssim"] = value["ssim"]
    return row


def main(args):
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_set = SRPatchDataset(
        args.train_dir,
        args.scale,
        args.lr_patch_size,
        args.channels,
        args.repeat,
        not args.no_augment,
        not args.no_paper_aug,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = FSRCNNDeconv(
        args.scale,
        args.d,
        args.s,
        args.m,
        args.channels,
        args.deconv_kernel,
        args.deconv_std,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    best_psnr = float("-inf")
    best_ssim = float("-inf")
    if args.resume:
        state, checkpoint = load_checkpoint(args.resume, device)
        assert_checkpoint_y_convention(
            checkpoint, allow_legacy=args.allow_legacy_checkpoint
        )
        model.load_state_dict(state, strict=True)
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        extra = checkpoint.get("extra", {})
        best_psnr = float(extra.get("best_psnr", best_psnr))
        best_ssim = float(extra.get("best_ssim", best_ssim))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "epochs").mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "model": "FSRCNNDeconv",
            "num_parameters": count_parameters(model),
            "degradation": "full-image Pillow bicubic before aligned patch extraction",
            "y_convention": Y_CONVENTION,
            "metric_peak": 1.0,
            "reconstruction_init": f"normal(0,{args.deconv_std})",
        }
    )

    print(f"Device       : {device}")
    print(
        f"Model        : FSRCNN Deconv x{args.scale}, d={args.d}, "
        f"s={args.s}, m={args.m}, channels={args.channels}"
    )
    print(f"Parameters   : {count_parameters(model):,}")
    print(f"Train dir    : {args.train_dir} ({len(train_set.paths)} images)")
    print(f"Validation   : {', '.join(args.val_dirs)}")
    print(f"Save dir     : {save_dir}")
    print(f"Y convention : {Y_CONVENTION if args.channels == 1 else 'RGB [0,1]'}")
    print("Loss/optim   : L1 + Adam")
    print("Degradation  : complete HR -> bicubic LR -> aligned LR/HR crops")
    print(f"Deconv init  : Gaussian mean=0, std={args.deconv_std}")

    amp_enabled = args.amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        def autocast_context():
            return torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        def autocast_context():
            return torch.cuda.amp.autocast(enabled=amp_enabled)

    last_metrics = {}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for lr, hr in progress:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                sr = model(lr)
                if sr.shape != hr.shape:
                    raise RuntimeError(
                        f"Training shape mismatch: SR={tuple(sr.shape)}, HR={tuple(hr.shape)}"
                    )
                loss = F.l1_loss(sr, hr)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        average_loss = loss_sum / max(len(train_loader), 1)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            last_metrics = evaluate(
                model, args.val_dirs, args.scale, args.channels, device
            )
            mean_psnr = mean_metric(last_metrics, "psnr")
            mean_ssim = mean_metric(last_metrics, "ssim")
            print(
                f"Epoch {epoch:04d} | loss={average_loss:.6f} | "
                f"{format_metrics(last_metrics)} | "
                f"Mean validation: PSNR={mean_psnr:.3f} dB, SSIM={mean_ssim:.4f}"
            )

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                extra = {
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "metrics": last_metrics,
                    "train_loss": average_loss,
                }
                save_checkpoint(
                    save_dir / "best_psnr.pth", model, optimizer, epoch, config, extra
                )
                save_checkpoint(
                    save_dir / "best.pth", model, optimizer, epoch, config, extra
                )

            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                save_checkpoint(
                    save_dir / "best_ssim.pth",
                    model,
                    optimizer,
                    epoch,
                    config,
                    {
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "metrics": last_metrics,
                        "train_loss": average_loss,
                    },
                )

            append_metrics_csv(
                save_dir / "metrics.csv",
                csv_row(epoch, average_loss, last_metrics, best_psnr, best_ssim),
            )
        else:
            print(f"Epoch {epoch:04d} | loss={average_loss:.6f}")

        extra = {
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "metrics": last_metrics,
            "train_loss": average_loss,
        }
        save_checkpoint(
            save_dir / "epochs" / f"epoch_{epoch:04d}.pth",
            model,
            optimizer,
            epoch,
            config,
            extra,
        )
        save_checkpoint(
            save_dir / "latest.pth", model, optimizer, epoch, config, extra
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PyTorch FSRCNN with MATLAB-compatible limited-range Y."
    )
    parser.add_argument("--train-dir", default="data/91-image")
    parser.add_argument(
        "--val-dirs", nargs="+", default=["data/Set5", "data/Set14"]
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--d", type=int, default=56)
    parser.add_argument("--s", type=int, default=12)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--deconv-kernel", type=int, default=9)
    parser.add_argument("--deconv-std", type=float, default=0.001)
    parser.add_argument("--lr-patch-size", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-paper-aug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-dir", default="runs/fsrcnn_deconv_x2")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
