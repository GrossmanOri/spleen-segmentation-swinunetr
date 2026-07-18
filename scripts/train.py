import os, sys, site, json, time, math, glob, argparse
us = site.getusersitepackages()          # make ~/.local monai importable inside the container
if us not in sys.path:
    sys.path.insert(0, us)

import torch
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, AsDiscrete)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", required=True)
    p.add_argument("--data_dir", default="/home/ori.grossman/nn_final/data/Task09_Spleen")
    p.add_argument("--out_root", default="/home/ori.grossman/nn_final/experiments")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--feature_size", type=int, default=48)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = get_args()
    set_determinism(seed=a.seed)
    device = torch.device("cuda")
    out_dir = os.path.join(a.out_root, a.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    print("config:", vars(a), flush=True)

    imgs = sorted(f for f in glob.glob(a.data_dir + "/imagesTr/*.nii.gz") if not os.path.basename(f).startswith("._"))
    lbls = sorted(f for f in glob.glob(a.data_dir + "/labelsTr/*.nii.gz") if not os.path.basename(f).startswith("._"))
    data = [{"image": i, "label": l} for i, l in zip(imgs, lbls)]
    train_files, val_files = data[:-9], data[-9:]

    ct_min, ct_max, pixdim, patch = -57, 164, (1.5, 1.5, 2.0), (96, 96, 96)
    base = [LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=ct_min, a_max=ct_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image")]
    rand = [RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=patch,
            pos=1, neg=1, num_samples=a.num_samples, image_key="image", image_threshold=0)]
    if a.augment:
        rand += [RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
                 RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
                 RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
                 RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
                 RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5)]
    train_tf, val_tf = Compose(base + rand), Compose(base)

    train_ds = CacheDataset(data=train_files, transform=train_tf, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_tf, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=4, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    model = SwinUNETR(img_size=patch, in_channels=1, out_channels=2,
                      feature_size=a.feature_size, use_checkpoint=True).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_label = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)

    def lr_factor(ep):
        if a.scheduler == "none":
            return 1.0
        if ep < a.warmup:
            return (ep + 1) / max(1, a.warmup)
        t = (ep - a.warmup) / max(1, a.epochs - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)

    best_dice, best_epoch, t0 = -1.0, -1, time.time()
    hist = {"config": vars(a), "epoch_loss": [], "val_epochs": [], "val_dice": [], "lr": []}
    for epoch in range(1, a.epochs + 1):
        model.train()
        ep_loss, steps = 0.0, 0
        for b in train_loader:
            x, y = b["image"].to(device), b["label"].to(device)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            steps += 1
        ep_loss /= steps
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        hist["epoch_loss"].append(ep_loss)
        hist["lr"].append(cur_lr)
        line = f"epoch {epoch:3d}/{a.epochs}  loss {ep_loss:.4f}  lr {cur_lr:.2e}"
        if epoch % a.val_interval == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for vb in val_loader:
                    vx, vy = vb["image"].to(device), vb["label"].to(device)
                    with torch.autocast("cuda"):
                        vout = sliding_window_inference(vx, patch, 4, model, overlap=0.5)
                    dice_metric(y_pred=[post_pred(o) for o in decollate_batch(vout)],
                                y=[post_label(o) for o in decollate_batch(vy)])
            dice = dice_metric.aggregate().item()
            hist["val_epochs"].append(epoch)
            hist["val_dice"].append(dice)
            line += f"  |  val Dice {dice:.4f}"
            if dice > best_dice:
                best_dice, best_epoch = dice, epoch
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
                line += "  <- best"
            hist["best_dice"], hist["best_epoch"] = best_dice, best_epoch
            json.dump(hist, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
        print(line, flush=True)

    hist["best_dice"], hist["best_epoch"], hist["minutes"] = best_dice, best_epoch, (time.time() - t0) / 60
    json.dump(hist, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print(f"\nDONE [{a.exp_name}] {hist['minutes']:.1f} min | best Dice {best_dice:.4f} @ {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
