import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from medpy.metric.binary import hd95  # For HD95 computation
from metrics import compute_dice, compute_hd95


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, sample in enumerate(dataloader):  # Access sample directly
        X = sample['image'].to(device)  # Input images (shape: [batch_size, 1, 224, 224])
        y = sample['label'].to(device)  # Ground truth labels (shape: [batch_size, 224, 224])
        
        # Forward pass
        pred = model(X)
        if isinstance(pred, list):  # If model outputs multiple stages
            pred = pred[-1] 
        loss = loss_fn(pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)  # Average training loss

# Validation loop
def validate_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    dice_scores = {1: [], 2: [], 3: [], 4: [], 5: []}  # Class indices for LV, RV, MYO
    hd95_scores = {1: [], 2: [], 3: [], 4: [], 5: []}  # Class indices for LV, RV, MYO

    with torch.no_grad():
        for sample in dataloader:  # Access sample directly
            X = sample['image'].to(device)  # Input images (shape: [batch_size, 1, 224, 224])
            y = sample['label'].to(device)  # Ground truth labels (shape: [batch_size, 224, 224])
              
            pred = model(X)
            if isinstance(pred, list):  # If model outputs multiple stages
                pred = pred[-1] 
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            # Compute metrics per class
            pred_classes = torch.argmax(pred, dim=1)  # Convert logits to class predictions
            y_np = y.cpu().numpy()
            pred_np = pred_classes.cpu().numpy()

            for c in [1, 2, 3, 4, 5]:  # LV, RV, MYO
                dice = compute_dice(y_np, pred_np, c)
                dice_scores[c].append(dice)

                hd = compute_hd95(y_np, pred_np, c)
                if hd is not None:  # Avoid invalid cases
                    hd95_scores[c].append(hd)

    # Average metrics
    avg_dice = {c: np.mean(dice_scores[c]) for c in dice_scores}
    avg_hd95 = {c: np.mean(hd95_scores[c]) if hd95_scores[c] else None for c in hd95_scores}

    return total_loss / len(dataloader), avg_dice, avg_hd95