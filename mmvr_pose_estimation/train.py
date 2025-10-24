import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.lenet import LeNet
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import MMVR


def get_max_preds(heatmaps):
    # heatmaps: (B, K, H, W)
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
    preds = torch.zeros((B, K, 2), device=heatmaps.device)
    preds[..., 0] = (idx % W).float()  # x
    preds[..., 1] = (idx // W).float()  # y
    return preds, maxvals


def train_model(
    model,
    dataloader_train,
    test_dataloader=None,
    num_epochs=50,
    lr=1e-3,
    device="cuda",
    mixed_precision=True,
    save_path="models/lenet.pt",
):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = GradScaler(enabled=mixed_precision)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        has_output_sample = False
        model.train()
        train_loss = 0.0

        pbar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            inputs, targets, _ = batch

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            with autocast(device, enabled=mixed_precision):
                outputs = model(inputs)
                loss = ((outputs - targets) ** 2 * (1 + 5 * targets)).mean()

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / len(dataloader_train)

        # ------------------ Validation ------------------
        if test_dataloader is not None:
            model.train()
            val_loss = 0.0
            total_correct = 0
            total_keypoints = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, targets, kps = batch
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    kps = kps.to(device, non_blocking=True)

                    with autocast(device, enabled=mixed_precision):
                        outputs = model(inputs)
                        loss = ((outputs - targets) ** 2 * (1 + 5 * targets)).mean()
                    val_loss += loss.item()

                    pred_coords, _ = get_max_preds(outputs)
                    pred_coords *= 10  # Output heatmaps are scaled by 1/10th
                    gt_coords = kps[:, :, :2]  # Drop confidence

                    dists = torch.norm(pred_coords - gt_coords, dim=2)
                    threshold = max(outputs.shape[2], outputs.shape[3])
                    correct = (dists < threshold).float()
                    total_correct += correct.sum().item()
                    total_keypoints += correct.numel()

            avg_val_loss = val_loss / len(test_dataloader)
            pck = 100 * total_correct / total_keypoints
            print(f"PCK: {pck}")

            if not has_output_sample:
                np.savez(
                    "sample-output.npz",
                    guess=pred_coords.cpu().numpy(),
                    truth=gt_coords.cpu().numpy(),
                )
                has_output_sample = True
        else:
            avg_val_loss = float("nan")

        # ------------------ Logging ------------------
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if test_dataloader is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (val_loss={avg_val_loss:.4f})")

    print("Training complete.")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
def get_dataset(file_path: str):
    npz = np.load(file_path)
    return MMVR(npz["X"], npz["y"], npz["kp"])


if __name__ == "__main__":
    train_ds = get_dataset("data/processed/d1s2-train.npz")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    gc.collect()

    test_ds = get_dataset("data/processed/d1s2-test.npz")
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # Instantiate model and train
    model = LeNet()
    train_model(model, train_dl, test_dl, num_epochs=20)
