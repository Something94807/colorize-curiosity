import argparse
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

IMG_SIZE = 512

def lab_to_rgb(l_tensor, ab_tensor):
    l = ((l_tensor.squeeze(0).cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0).numpy().astype(np.uint8)
    ab = ((ab_tensor.cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0).numpy().astype(np.uint8)
    lab_arr = np.transpose(np.vstack([l[None, ...], ab]), (1, 2, 0))
    return Image.fromarray(lab_arr, mode="LAB").convert("RGB")

class MarsColorDataset(Dataset):
    def __init__(self, data_dir: str, filenames: list, img_size: int = IMG_SIZE, is_train: bool = True):
        self.color_dir = os.path.join(data_dir, "color")
        self.gray_dir = os.path.join(data_dir, "gray")
        self.filenames = filenames
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        color_img = Image.open(os.path.join(self.color_dir, fname)).convert("RGB")
        gray_img = Image.open(os.path.join(self.gray_dir, fname)).convert("L")

        if self.is_train:
            load_size = int(self.img_size * 1.12)
            color_img = TF.resize(color_img, [load_size, load_size])
            gray_img = TF.resize(gray_img, [load_size, load_size])

            i, j, h, w = transforms.RandomCrop.get_params(
                color_img, output_size=(self.img_size, self.img_size)
            )
            color_img = TF.crop(color_img, i, j, h, w)
            gray_img = TF.crop(gray_img, i, j, h, w)

            if random.random() > 0.5:
                color_img = TF.hflip(color_img)
                gray_img = TF.hflip(gray_img)

            gray_img = transforms.ColorJitter(brightness=0.2, contrast=0.3)(gray_img)
        else:
            color_img = TF.resize(color_img, [self.img_size, self.img_size])
            gray_img = TF.resize(gray_img, [self.img_size, self.img_size])

        lab_img = color_img.convert("LAB")
        lab_arr = np.array(lab_img, dtype=np.float32)

        ab_arr = (lab_arr[:, :, 1:] / 255.0) * 2.0 - 1.0
        ab_tensor = torch.from_numpy(ab_arr).permute(2, 0, 1)

        gray_tensor = TF.to_tensor(gray_img)
        gray_tensor = TF.normalize(gray_tensor, mean=[0.5], std=[0.5])

        return gray_tensor, ab_tensor

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

def upconv_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = conv_block(1, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)
        self.down5 = conv_block(512, 512)
        self.down6 = conv_block(512, 512)

        self.up1 = upconv_block(512, 512, dropout=True)
        self.up2 = upconv_block(1024, 512, dropout=True)
        self.up3 = upconv_block(1024, 256)
        self.up4 = upconv_block(512, 128)
        self.up5 = upconv_block(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        out = self.final(torch.cat([u5, d1], dim=1))
        return torch.tanh(out)

def compute_loss(model, gray, ab_target, use_amp):
    with torch.amp.autocast('cuda', enabled=use_amp):
        pred_ab = model(gray)
        loss = torch.nn.functional.l1_loss(pred_ab, ab_target)
    return loss

@torch.no_grad()
def evaluate(model, val_loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for gray, ab_target in val_loader:
        gray, ab_target = gray.to(device), ab_target.to(device)
        loss = compute_loss(model, gray, ab_target, use_amp)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)

def color_accuracy_pct(pred_img, real_img):
    lab_pred = np.array(pred_img.convert("LAB"), dtype=np.float32)
    lab_real = np.array(real_img.convert("LAB"), dtype=np.float32)
    ab_diff = lab_pred[..., 1:] - lab_real[..., 1:]
    dist = np.sqrt((ab_diff ** 2).sum(axis=-1))
    max_dist = np.sqrt(2 * 255 ** 2)
    return 100 * (1 - dist.mean() / max_dist)

def save_sample_grid(model, dataset, device, out_path, n=4):
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    rows = []
    caption_h = 20
    with torch.no_grad():
        for i in idxs:
            gray_tensor, ab_target = dataset[i]
            pred_ab = model(gray_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

            gray_img = Image.fromarray(
                ((gray_tensor.squeeze(0) * 0.5 + 0.5) * 255).numpy().astype(np.uint8), mode="L"
            ).convert("RGB")

            pred_img = lab_to_rgb(gray_tensor, pred_ab)
            real_img = lab_to_rgb(gray_tensor, ab_target)

            score = color_accuracy_pct(pred_img, real_img)

            row = Image.new("RGB", (gray_img.width * 3 + 20, gray_img.height + caption_h), "white")
            row.paste(gray_img, (0, 0))
            row.paste(pred_img, (gray_img.width + 10, 0))
            row.paste(real_img, (gray_img.width * 2 + 20, 0))
            draw = ImageDraw.Draw(row)
            draw.text((gray_img.width + 10, gray_img.height + 2), f"Color accuracy: {score:.1f}%", fill="black")
            rows.append(row)

    grid = Image.new("RGB", (rows[0].width, rows[0].height * len(rows) + 10 * (len(rows) - 1)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + 10
    grid.save(out_path)
    model.train()

def find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(r"model_epoch_(\d+)\.pt$")
    ckpts = [f for f in os.listdir(ckpt_dir) if pattern.match(f)]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(pattern.match(f).group(1)))
    return os.path.join(ckpt_dir, ckpts[-1])

def save_checkpoint(path, epoch, model, optimizer, scaler, best_val_loss, epochs_no_improve):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="mars_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    color_dir = os.path.join(args.data_dir, "color")
    if not os.path.exists(color_dir):
        raise RuntimeError(f"Directory {color_dir} not found.")

    all_files = sorted(os.listdir(color_dir))
    if not all_files:
        raise RuntimeError(f"No images found in {color_dir}.")

    random.seed(42)
    random.shuffle(all_files)

    val_count = max(1, int(0.1 * len(all_files)))
    train_files = all_files[val_count:]
    val_files = all_files[:val_count]

    train_dataset = MarsColorDataset(args.data_dir, train_files, img_size=IMG_SIZE, is_train=True)
    val_dataset = MarsColorDataset(args.data_dir, val_files, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Loaded {len(train_files)} training pairs and {len(val_files)} validation holdout pairs.")

    model = UNetColorizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "model_best.pt")

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0
    if args.resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            print(f"Resuming from {latest}")
            checkpoint = torch.load(latest, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            print(f"Resuming training at epoch {start_epoch} (best val loss: {best_val_loss:.4f})")
        else:
            print("No checkpoint found in checkpoints/, starting from scratch.")

    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        for gray, ab_target in train_loader:
            gray, ab_target = gray.to(device), ab_target.to(device)

            optimizer.zero_grad()
            loss = compute_loss(model, gray, ab_target, use_amp)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} - Train loss: {avg_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            val_loss = evaluate(model, val_loader, device, use_amp)
            print(f"  Val loss: {val_loss:.4f}")

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, best_val_loss, epochs_no_improve)
            print(f"  Saved checkpoint -> {ckpt_path}")

            if improved:
                save_checkpoint(best_ckpt_path, epoch, model, optimizer, scaler, best_val_loss, epochs_no_improve)
                print(f"  New best val loss -> saved {best_ckpt_path}")

            sample_path = os.path.join(sample_dir, f"epoch_{epoch}.png")
            save_sample_grid(model, val_dataset, device, sample_path)
            print(f"  Saved sample grid -> {sample_path}")

            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(f"No val improvement in {epochs_no_improve} save-intervals. Stopping early at epoch {epoch}.")
                break

    print("Training complete.")

if __name__ == "__main__":
    main()