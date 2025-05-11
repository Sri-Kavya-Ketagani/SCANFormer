import torch
import torch.nn as nn
import torch.nn.functional as F

#Loss Function
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight  # Weight for Cross-Entropy Loss
        self.dice_weight = dice_weight  # Weight for Dice Loss
    
    def forward(self, outputs, labels):
        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(outputs, labels)

        # Compute Dice Loss (one-hot encoding is done within dice_loss)
        dice_loss = self.dice_loss(outputs, labels)

        # Combine both losses
        total_loss = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)
        return total_loss
    
    def dice_loss(self, outputs, labels, smooth=1e-6):

        num_classes = outputs.size(1)  # Number of classes
        device = outputs.device

        # One-hot encode labels to shape [B, C, H, W]
        labels_one_hot = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float().to(device)

        # Convert outputs to probabilities (softmax)
        probs = F.softmax(outputs, dim=1)

        # Flatten for Dice computation
        probs = probs.contiguous().view(outputs.size(0), num_classes, -1)
        labels_one_hot = labels_one_hot.contiguous().view(outputs.size(0), num_classes, -1)

        # Compute intersection and union
        intersection = (probs * labels_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + labels_one_hot.sum(dim=2)

        # Compute Dice score
        dice_score = (2.0 * intersection + smooth) / (union + smooth)

        # Dice Loss is 1 - mean Dice score across all classes
        dice_loss = 1 - dice_score.mean()
        return dice_loss