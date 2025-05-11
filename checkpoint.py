import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define checkpoint directory
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, scheduler, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"✅ Checkpoint saved at epoch {epoch}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(os.path.join(checkpoint_dir, filename), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"🔄 Resuming training from epoch {start_epoch}...")
    return start_epoch