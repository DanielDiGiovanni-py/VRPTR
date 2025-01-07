# VRPTR: Predicting Task-Activated fMRI from Resting-State Data on Cortical Meshes

**Author**: Daniel A. Di Giovanni
**Chapters Referenced**: Chapter 2 (“Resting-State Functional Connectivity for Brain Mapping”) and Chapter 5 (“Mesh-Based Deep Learning Models for fMRI Analysis”)

---

## Overview

This repository contains code for **VRPTR** (“Variational Resting-state to Task Prediction Transformer”), a deep learning framework to predict **task-evoked** fMRI activity from **resting-state** fMRI data on cortical surfaces. The project is inspired by concepts and methodologies introduced in **Chapter 2** (fundamentals of resting-state functional connectivity in neurosurgical planning) and **Chapter 5** (mesh-based CNNs and Transformers for fMRI data) of the attached thesis.

### What is VRPTR?

VRPTR leverages **surface-based** neural networks (U-Net style) combined with a **Transformer-based** latent space and **variational** (VAE-like) regularization. The approach addresses two core challenges:

1. **Spatial Complexity**: The cortex is represented as a triangular mesh (icospheres), capturing detailed geometry rather than forcing a voxel grid.  
2. **Functional Variability**: Resting-state functional connectivity is leveraged to infer potential **task activations**—helpful in settings where actual task fMRI may be missing, impractical, or risky (e.g., preoperative planning).

---

## Motivation and Background (Chapter 2)

According to **Chapter 2** of the thesis, resting-state networks have become increasingly pivotal in neurosurgical planning when performing **functional localization**. Key highlights that motivate VRPTR include:

- **Non-Invasive, No Task Required**: Resting-state scans reduce patient burden and time.  
- **Potential for Preoperative Mapping**: Surgeons can glean an approximation of activation patterns (e.g., motor, language) from resting-state connectivity alone.  
- **Challenges**: Mapping from “connectivity fingerprints” to actual task-activated patterns is highly **non-linear** and requires large-scale data-driven approaches—hence the impetus for a deep learning solution.

**(Figure)** Below is an illustrative figure from **Chapter 2** showing resting-state connectivity patterns for a subject (adapted from the thesis).

![Figure from Chapter 2: Resting-state networks](images/chapter2-rsfc.png)  
*Figure 2.1. Example resting-state networks identified in the cortex. Green outlines a sensorimotor network, and orange outlines a language network (adapted from the thesis).*

---

## How VRPTR Works (Chapter 5)

**Chapter 5** of the thesis outlines a **mesh-based** convolutional neural network with **Transformer** layers. The VRPTR pipeline is as follows:

1. **Data Representation**:  
   - Cortical surfaces (left and right hemispheres) are represented as **icosphere meshes** with progressively coarser or finer resolutions (e.g., “icosphere_2.pkl”, “icosphere_3.pkl”).  
   - **Resting-state** data is stored as a matrix of size `[\text{channels}, \text{vertices}]`, where channels might be “principal components” or “RSFC measures” (e.g., correlation to principal seeds).

2. **U-Net Encoder-Decoder**:
   - **Encoder**: A downsampling path with mesh-specific pooling (moving from a finer mesh to coarser meshes).  
   - **Decoder**: An upsampling path using transpose mesh convolutions that reconstruct to the original resolution.  
   - **Skip Connections**: Retain details from earlier layers (finer resolutions) in typical U-Net style.

3. **Transformer Bottleneck**:
   - In the coarsest latent space, VRPTR inserts a **Transformer** with multi-head self-attention.  
   - This step captures long-range dependencies on the cortical surface, hypothesized to be crucial for functional connectivity patterns.

4. **Variational Regularization**:
   - VRPTR includes a **VAE-like** branch with learned **\(\mu\)** and **\(\sigma\)**.  
   - The **KL divergence** is tracked as a regularization term in training. This helps the model learn a smoother, more generalizable latent space.

5. **Output**: Predicted **task-activation** maps (e.g., for motor/language tasks) at each vertex, derived purely from the resting-state input.

Below is a conceptual figure from **Chapter 5** illustrating VRPTR’s architecture.

![Figure from Chapter 5: VRPTR Architecture](images/chapter5-vrptr.png)  
*Figure 5.2. The VRPTR pipeline. A U-Net with mesh-based convolutions is augmented by a Transformer-based latent space and a VAE-like reparameterization step (adapted from the thesis).*

---

## Repository Structure

This codebase is split into four main scripts:

| **File**              | **Description**                                                                                               |
|-----------------------|---------------------------------------------------------------------------------------------------------------|
| `vrptr_train.py`      | Script for **training** the VRPTR model (loads data, creates DataLoader, runs epochs, saves checkpoints).     |
| `vrptr_dataset.py`    | Defines the **VRPTRDataset** class (PyTorch `Dataset`) to load resting-state and task-contrast .npy files.    |
| `vrptr_model.py`      | Contains the **RPTR** model class (U-Net + Transformer + VAE) and supporting mesh-convolution modules.       |
| `vrptr_test.py`       | Script for **testing** or inference—loads a trained model checkpoint, applies it to new data, saves outputs. |

Additionally:

- `checkpoints/` (created automatically) will store model `.pth` files.  
- `logs/` (created automatically) will store training logs.  
- `predictions/` or `vrptr_outputs/` might store test-time predictions.  
- `data/` directory should hold your mesh files (e.g., `icosphere_2.pkl`, etc.) and the `.npy` data for RSFC and task contrasts.

---

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourLab/VRPTR.git
   cd VRPTR
   ```
2. **Install Requirements** (assuming Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
   Example packages:
   - `numpy`
   - `scipy`
   - `torch`
   - `torchvision`
   - `scikit-learn`
   - `nibabel` (optional if using NiFTI data)
   - `matplotlib` (for potential plotting)

3. **Data Preparation**:
   - Ensure your **cortical mesh** files (`icosphere_{level}.pkl`, etc.) are in `data/fs_LR_mesh_templates` or a chosen directory.  
   - Place your **resting-state** and **task contrast** `.npy` files in a structure consistent with how the dataset scripts expect them:
     ```
     data/
       rsfc_d50_sample1/
         joint_LR_SUBJECTID_sample0_rsfc.npy
         joint_LR_SUBJECTID_sample1_rsfc.npy
         ...
       ts_obj/
         SUBJECTID_joint_LR_task_contrasts.npy
       subj_tumor.txt  (list of subject IDs)
     ```

---

## Usage

### Training

```bash
python vrptr_train.py \
    --experiment VRPTR_experiment_01 \
    --root /path/to/data \
    --rsfc_dir rsfc_d50_sample1 \
    --contrast_dir ts_obj \
    --subj_list subj_tumor.txt \
    --lr 1e-2 \
    --batch_size 4 \
    --epochs 2000 \
    --save_freq 200 \
    --gpu 0
```

- A new directory `checkpoints/VRPTR_experiment_01` will store:
  - `model_epoch_{epoch}.pth` (intermediate checkpoints every `--save_freq` epochs)
  - `model_epoch_last.pth` (final checkpoint)

**Log messages** and training stats appear in `logs/VRPTR_experiment_01.log`.

### Testing / Inference

```bash
python vrptr_test.py \
    --checkpoint checkpoints/VRPTR_experiment_01/model_epoch_1999.pth \
    --root /path/to/data \
    --rsfc_dir rsfc_d50_sample1 \
    --subj_list subj_tumor.txt \
    --output_dir vrptr_outputs \
    --num_samples 1 \
    --gpu 0
```

- Produces `.npy` predictions in `vrptr_outputs/SUBJECTID_pred.npy`.
- If `--num_samples` > 1, it will process multiple resting-state samples per subject and stack the results.

---

## Potential Use Cases

1. **Preoperative Mapping**: Predict activation for critical functions (motor, language) when actual task fMRI is unavailable or too risky.  
2. **Rapid Screening**: Evaluate patient cohorts on resting-state scans for specific functional deficits or reorganization patterns.  
3. **Research in FC-Task Relationship**: Investigate how resting-state networks translate into task-evoked patterns across large subject datasets.

---

## Limitations and Future Directions

- **Data Quantity**: The model relies on having sufficient training subjects with both resting-state and task-based data.  
- **Resolution**: The current icosphere mesh resolution (e.g., level 2 or 3) may limit fine-grained localization.  
- **Transformer Overhead**: Adding Transformers to a U-Net can be GPU/memory-intensive, particularly at higher resolutions.

For more details on these aspects, refer to **Chapter 5** of the thesis, which discusses solutions such as partial convolution, block-wise training, or advanced regularization strategies.

---

## References

1. **Thesis**: [Daniel A. Di Giovanni], *Deep Learning and Statistical Methods for Clinical Application of
Functional Magnetic Resonance Imaging*, 2024, Chapter 2, Chapter 5. (https://escholarship.mcgill.ca/downloads/s4655p01n)
2. [SabuncuLab BrainSurfCNN Repo](https://github.com/sabunculab/brainsurfcnn)  
3. [UGSCNN Repo](https://github.com/maxjiang93/ugscnn)

---

*© 2025, VRPTR Project. All rights reserved.*
