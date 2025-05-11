from medpy.metric.binary import hd95
import numpy as np

# Helper function: Dice coefficient
def compute_dice(y_true, y_pred, class_idx):
    y_true_c = (y_true == class_idx).astype(np.float32)
    y_pred_c = (y_pred == class_idx).astype(np.float32)

    intersection = np.sum(y_true_c * y_pred_c)
    union = np.sum(y_true_c) + np.sum(y_pred_c)
    return 2 * intersection / (union + 1e-6)

# Helper function: HD95
def compute_hd95(y_true, y_pred, class_idx):
    y_true_c = (y_true == class_idx).astype(np.uint8)
    y_pred_c = (y_pred == class_idx).astype(np.uint8)

    if np.any(y_true_c) and np.any(y_pred_c):  # Both ground truth and prediction exist
        return hd95(y_true_c, y_pred_c)
    return None