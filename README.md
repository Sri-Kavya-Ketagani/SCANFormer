# SCANFormer: Scale and Context-Aware Nested Feature Fusion Transformer for Medical Image Segmentation

This repository contains the official implementation of **SCANFormer**, a hybrid CNN-Transformer model designed for accurate and efficient medical image segmentation.

## Environment Setup

To replicate our experiments, please use:

- Python 3.10.12
- PyTorch 2.5.1

Install all required dependencies using:

```bash
pip install -r requirements.txt
```


## Train/Test Instructions
Before training, please configure the dataset path in train.py. You can adjust other parameters such as batch_size, epochs, etc., as needed.

Run the training script:

```bash
python train.py
```

Similarly for testing, configure the test dataset and output_dir paths to save model predictions and then run:

```bash
python test.py
```

## Codes used in our experiments and Acknowledgements
[Attention U-Net](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)

[UNet++](https://github.com/qubvel-org/segmentation_models.pytorch) 

[TransUNet](https://github.com/Beckschen/TransUNet)

[Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)
