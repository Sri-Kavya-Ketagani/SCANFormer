import os
import random

# Load all slices from the file
with open("ACDC/ACDC/all_slices.list", "r") as file:
    all_slices = file.readlines()

# Strip any trailing whitespace (like newlines)
all_slices = [line.strip() for line in all_slices]

# Shuffle the data to ensure random selection
random.shuffle(all_slices)

# Calculate split sizes
total_slices = len(all_slices)
train_size = int(0.7 * total_slices)
val_size = int(0.1 * total_slices)
test_size = total_slices - train_size - val_size  # Remaining for test

# Split into train, validation, and test sets
train_slices = all_slices[:train_size]
val_slices = all_slices[train_size:train_size + val_size]
test_slices = all_slices[train_size + val_size:]

os.makedirs("ACDC/splits", exist_ok=True)

# Save the splits into separate files
with open("ACDC/splits/train_slices.list", "w") as train_file:
    train_file.write("\n".join(train_slices))

with open("ACDC/splits/val_slices.list", "w") as val_file:
    val_file.write("\n".join(val_slices))

with open("ACDC/splits/test_slices.list", "w") as test_file:
    test_file.write("\n".join(test_slices))

print(f"Total slices: {total_slices}")
print(f"Train: {len(train_slices)}, Validation: {len(val_slices)}, Test: {len(test_slices)}")