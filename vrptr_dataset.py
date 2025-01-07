#!/usr/bin/env python3
"""
VRPTR Dataset

Defines the VRPTRDataset class to load resting-state fMRI data (rsFC)
and corresponding task-contrast maps. 
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class VRPTRDataset(Dataset):
    """
    A PyTorch Dataset for VRPTR:
      - subj_ids: list or array of subject IDs
      - rsfc_dir: directory containing rsFC .npy files
      - contrast_dir: directory containing task contrast .npy files
      - num_samples: if each subject has multiple (bootstrap) resting-state samples
    """
    def __init__(self, subj_ids, rsfc_dir, contrast_dir, num_samples=8):
        super().__init__()
        self.subj_ids = subj_ids
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.num_samples = num_samples

    def __len__(self):
        return len(self.subj_ids)

    def __getitem__(self, index):
        """
        Returns:
          rsfc_data: [in_ch, #vertices] as a FloatTensor
          task_data: [out_ch, #vertices] as a FloatTensor
        """
        subj = self.subj_ids[index]
        
        # Pick a random sample among num_samples
        sample_id = np.random.randint(0, self.num_samples)
        rsfc_file = os.path.join(self.rsfc_dir,
                                 f"joint_LR_{subj}_sample{sample_id}_rsfc.npy")
        task_file = os.path.join(self.contrast_dir,
                                 f"{subj}_joint_LR_task_contrasts.npy")
        
        rsfc_data = np.load(rsfc_file)      # shape: [in_ch, #vertices]
        task_data = np.load(task_file)      # shape: [out_ch, #vertices] or similar

        # Convert to Torch tensors on CPU here (move to GPU in train/test loops)
        rsfc_tensor = torch.from_numpy(rsfc_data).float()
        task_tensor = torch.from_numpy(task_data).float()
        
        return rsfc_tensor, task_tensor
