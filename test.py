import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fvcore.nn import FlopCountAnalysis
from medpy.metric.binary import hd95
import os
from tqdm import tqdm
from PIL import Image

# Load the model architecture first
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UDTransNet(config, n_channels=1, n_classes=4, img_size=224)  # Replace `YourModel` with U-Net, U-Net++, UDTransNet, etc.
model.to(device)

# Load the trained weights
checkpoint = torch.load('/kaggle/input/udtransnet/pytorch/default/1/final_model.pth', map_location=device)
model.load_state_dict(checkpoint)

# Set to evaluation mode
model.eval()

# Define directory to save visualizations
save_dir = "./predictions"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists



class_colors = ['black', 'red', 'green', 'blue']
cmap = mcolors.ListedColormap(class_colors)

def overlay_mask(ax, image, mask, alpha=0.6, cmap=cmap):
    """ Overlays a segmentation mask on an image while keeping grayscale intact. """
    
    # Convert grayscale image to 3-channel RGB to preserve original contrast
    image_rgb = np.stack([image] * 3, axis=-1)  # Convert [H, W] → [H, W, 3]
    
    # Display the grayscale image in the background
    ax.imshow(image_rgb, cmap='gray', vmin=0, vmax=1)

    # Create mask overlay only for nonzero pixels (ignore background)
    masked_overlay = np.ma.masked_where(mask == 0, mask)  # Ignore background (0)
    
    # Overlay only segmented regions (not the full image)
    im = ax.imshow(masked_overlay, cmap=cmap, alpha=alpha, vmin=0, vmax=len(class_colors)-1)
    
    ax.axis('off')
    return im

# Loop through all test samples
for i in range(len(test_loader)):
    sample_batch = next(iter(test_loader))  # Get batch
    images = sample_batch['image'].to(device)
    masks = sample_batch['label'].to(device)

    with torch.no_grad():
        predictions = model(images)
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()

    gt_classes = masks.cpu().numpy() if len(masks.shape) == 3 else torch.argmax(masks, dim=1).cpu().numpy()

    # Create subplots
    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))

    if len(images) == 1:  # Fix indexing issue when there's only 1 image
        axes = np.expand_dims(axes, axis=0)  # Ensure it's 2D

    for idx in range(len(images)):
        image = images[idx, 0, :, :].cpu().numpy().squeeze()  # Ensure it's 2D
        gt_mask = gt_classes[idx]  # Ground truth mask
        pred_mask = pred_classes[idx]  # Predicted mask

        # Normalize image for better visualization (optional)
        image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1

        # Plot input image (ensure grayscale is displayed properly)
        axes[idx, 0].imshow(image, cmap='gray', vmin=0, vmax=1)  
        axes[idx, 0].set_title("Input Image")
        axes[idx, 0].axis('off')

        # Overlay ground truth mask on the image (fixed)
        im = overlay_mask(axes[idx, 1], image, gt_mask, alpha=0.7, cmap=cmap)
        axes[idx, 1].set_title("Ground Truth Overlay")

        # Overlay predicted mask on the image (fixed)
        im = overlay_mask(axes[idx, 2], image, pred_mask, alpha=0.7, cmap=cmap)
        axes[idx, 2].set_title("Predicted Mask Overlay")

    plt.tight_layout()
    plt.show()


# Mean Performance
def compute_hd95(y_true, y_pred, class_idx):
    from medpy.metric.binary import hd95  # Ensure correct function is used
    y_true_c = (y_true == class_idx).astype(np.uint8)
    y_pred_c = (y_pred == class_idx).astype(np.uint8)

    if np.any(y_true_c) and np.any(y_pred_c):  # If both GT and Pred exist
        return float(hd95(y_true_c, y_pred_c))  # Ensure proper conversion
    return None

def compute_dice(y_true, y_pred, class_idx):
    y_true_c = (y_true == class_idx).astype(np.float32)
    y_pred_c = (y_pred == class_idx).astype(np.float32)

    intersection = np.sum(y_true_c * y_pred_c)
    union = np.sum(y_true_c) + np.sum(y_pred_c)
    return 2 * intersection / (union + 1e-6)

# Store metrics for all test samples
num_classes = 4
all_dice_scores = {cls: [] for cls in range(1, num_classes)}  # Exclude background (0)
all_hd95_scores = {cls: [] for cls in range(1, num_classes)}

# Loop through all test samples
for i, sample_batch in enumerate(test_loader):
    images = sample_batch['image'].to(device)
    masks = sample_batch['label'].to(device)

    with torch.no_grad():
        predictions = model(images)
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    
    gt_classes = masks.cpu().numpy() if len(masks.shape) == 3 else torch.argmax(masks, dim=1).cpu().numpy()

    # Compute Dice and HD95 for each class
    for cls in range(1, num_classes):  # Ignore background (0)
        dice = compute_dice(gt_classes[0], pred_classes[0], cls)
        hd = compute_hd95(gt_classes[0], pred_classes[0], cls)

        if dice is not None:
            all_dice_scores[cls].append(dice)
        if hd is not None:
            all_hd95_scores[cls].append(hd)

# Compute and print mean performance
print("\n### Mean Evaluation Metrics ###")
for cls in range(1, num_classes):
    mean_dice = np.mean(all_dice_scores[cls]) if all_dice_scores[cls] else 0
    mean_hd95 = np.mean(all_hd95_scores[cls]) if all_hd95_scores[cls] else 0
    print(f"Class {cls}: Mean Dice = {mean_dice:.4f}, Mean HD95 = {mean_hd95:.4f}")


# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the parameter count in millions
param_count = count_parameters(model)
print(f"Total number of parameters: {param_count / 1e6:.2f} M")



# FLOPS
random_input = torch.randn(1, 1, 224, 224).to(device)  # Adjust shape if needed
flops = FlopCountAnalysis(model, random_input)
print(f"FLOPs: {flops.total()/1e9:.2f} GFLOPs")
