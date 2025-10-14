import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.lenet import LeNet
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import MMVR


# -------------------------------------------------------------------------
# Example training loop
# -------------------------------------------------------------------------
def train_model(
    model,
    dataloader_train,
    dataloader_val=None,
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
    criterion = nn.MSELoss(reduction="mean")
    

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        has_output_sample = False
        model.train()
        train_loss = 0.0

        pbar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            # Unpack batch
            if len(batch) == 3:
                inputs, targets, valid_mask = batch
            else:
                inputs, targets = batch
                valid_mask = None

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            with autocast(device, enabled=mixed_precision):
                outputs = model(inputs)
                if valid_mask is not None:
                    loss = criterion(outputs * valid_mask, targets * valid_mask)
                else:
                    loss =  ((outputs - targets)**2 * (1 + 5 * targets)).mean()

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if not has_output_sample:
                np.savez("sample-output.npz", guess=outputs.cpu().detach().numpy(), truth=targets.cpu().numpy())
                has_output_sample = True

        scheduler.step()
        avg_train_loss = train_loss / len(dataloader_train)

        # ------------------ Validation ------------------
        if dataloader_val is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in dataloader_val:
                    if len(batch) == 3:
                        inputs, targets, valid_mask = batch
                    else:
                        inputs, targets = batch
                        valid_mask = None

                    if not has_output_sample:
                        print(inputs.dtype)
                    inputs = inputs.to(device).float()
                    targets = targets.to(device)
                    if valid_mask is not None:
                        valid_mask = valid_mask.to(device)

                    outputs = model(inputs)
                    if valid_mask is not None:
                        loss = criterion(outputs * valid_mask, targets * valid_mask)
                    else:
                        loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    

            avg_val_loss = val_loss / len(dataloader_val)
        else:
            avg_val_loss = float("nan")

        # ------------------ Logging ------------------
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if dataloader_val is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (val_loss={avg_val_loss:.4f})")

    print("Training complete.")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_npz = np.load("data/interm/d1s2.npz")
    train_ds = MMVR(train_npz["X"], train_npz["y"])
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    # val_npz = np.load("data/interm/d3s2.npz")
    # val_ds = MMVR(val_npz["X"], val_npz["y"])
    # val_dl = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=2)

    # Instantiate model and train
    model = LeNet()
    train_model(model, train_dl, num_epochs=5)
