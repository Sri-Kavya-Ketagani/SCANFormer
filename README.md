# SCANFormer: Scale and Context-Aware Nested Feature Fusion Transformer for Medical Image Segmentation

This repository contains the official implementation of **SCANFormer**, a hybrid CNN-Transformer model designed for accurate and efficient medical image segmentation. SCANFormer is a novel and generalized medical image segmentation framework with a comprehensive
multi-scale feature extraction and fusion approach to enable more diverse feature learning. Extensive experiments on datasets of CT and MRI modalities demonstrate the superiority of SCANFormer, achieving 3%, 8.48%, 11.6% and 11.06% higher Dice scores than TransUNet on the ACDC, Synapse, LCTSC and Brats-Africa2024 datasets, respectively even when trained from scratch without relying on pre-trained models.


## Datasets
SCANFormer is evaluated on the following publicly available medical imaging datasets:

ACDC - Automated Cardiac Diagnosis Challenge consists of 100 MRI cases for segmentation of cardiac structures. The dataset can be downloaded from https://www.creatis.insa-lyon.fr/Challenge/acdc/

Synapse -  Multi-Atlas Abdomen Labeling Challenge, MICCAI 2015, consists of 30 CT scans covering 8 abdominal organ. The dataset can be downloaded from https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

LCTSC - Lung CT Segmentation Challenge 2017 contains 60 CT scans of the thoracic region. The dataset can be downloaded from https://www.cancerimagingarchive.net/collection/lctsc/

Brats-Africa2024 - Brain tumor segmentation challenge on African population contains 60 MRI patient cases. The dataset can be downloaded from
https://www.cancerimagingarchive.net/collection/brats-africa/


Before running the training or testing scripts, ensure that the datasets (ACDC, Synapse, LCTSC) are downloaded and organized in the following structure (you can adjust this in the code if needed):
```
datasets/
├── ACDC/
│ ├── images/
│ └── labels/
├── Synapse/
│ ├── images/
│ └── labels/
└── LCTSC/
├── images/
└── labels/
└── Brats2024/
├── images/
└── labels/

</details>
```


## Environment Setup

To replicate our experiments, please use:

- Python 3.10.12
- PyTorch 2.5.1

## Requirements
Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

Although the datasets contain 3D volumes, we used **2D slices** for training and evaluation to stay consistent with prior work and benchmark models.

All images and masks were resized to:
- `224×224` for **ACDC** and **Synapse**
- `256×256` for **LCTSC**

For the **LCTSC dataset**:
- Original **DICOM images** and **RTStruct masks** were converted into **NumPy arrays**.
- If the original slice size did not match the target resolution, **zero-padding** was applied to preserve the aspect ratio before resizing.

All images were normalized using **per-image z-score normalization** (mean 0, std 1).

**Data augmentation** included:
- Random rotations
- Random flipping (horizontal/vertical)


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

## Evaluation

Evaluation was conducted on 2D slices using dataset-specific splits following the experimental setup of [TransUNet](https://github.com/Beckschen/TransUNet):

- **ACDC**:
  - 70 samples for training
  - 10 samples for validation
  - 20 samples for testing
  - **Metric**: Dice Similarity Coefficient (DSC) only

- **Synapse**:
  - 18 cases for training
  - 12 cases for testing
  - **Metrics**: Dice Similarity Coefficient (DSC), 95% Hausdorff Distance (HD95)

- **LCTSC**:
  - 36 samples for training
  - 12 samples for validation
  - 12 samples for testing
  - **Metrics**: Dice Similarity Coefficient (DSC), 95% Hausdorff Distance (HD95)
 
- **Brats-Africa2024**:
  - 48 samples for training
  - 12 samples for testing
  - **Metrics**: Dice Similarity Coefficient (DSC) only

All models were trained and evaluated under identical conditions per dataset.

### Ablation Study

An ablation study was performed on the **ACDC** dataset to evaluate the effectiveness of key architectural components in **SCANFormer**, specifically the feature extraction and integration mechanisms. Variants with and without attention modules and skip fusion were compared using the Dice score.


## Codes used in our experiments and Acknowledgements
[Attention U-Net](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)

[UNet++](https://github.com/qubvel-org/segmentation_models.pytorch) 

[TransUNet](https://github.com/Beckschen/TransUNet)

[Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)

## References
[1] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?" in IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018
doi: 10.1109/TMI.2018.2837502

[2] inzhong Yang, Greg Sharp, Harini Veeraraghavan, Wouter Van Elmpt, Andre Dekker, T Lustberg, and M Gooding. Data from lung ct segmentation challenge 2017 (lctsc). The Cancer Imaging Archive, 2017

[3] Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., ... & Zhou, Y. (2021). Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
