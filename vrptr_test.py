#!/usr/bin/env python3
"""
VRPTR Testing Script

Loads a trained RPTR model, runs inference on specified subjects,
and saves predictions as .npy.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from vrptr_model import RPTR
from vrptr_dataset import VRPTRDataset


def parse_test_arguments():
    parser = argparse.ArgumentParser(description="Test the VRPTR model.")
    parser.add_argument('--gpu', default='0', type=str, help='Which GPU to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint .pth')
    parser.add_argument('--output_dir', type=str, default='vrptr_outputs',
                        help='Directory to save prediction .npy files')
    parser.add_argument('--root', default='/data/', type=str,
                        help='Root directory for data')
    parser.add_argument('--rsfc_dir', default='rsfc_d50_sample1', type=str,
                        help='Directory with rsFC .npy files')
    parser.add_argument('--contrast_dir', default='ts_obj', type=str,
                        help='Directory with contrast .npy files (unused in test?)')
    parser.add_argument('--subj_list', default='subj_tumor.txt', type=str,
                        help='Text file listing subject IDs for inference')
    parser.add_argument('--num_samples', default=1, type=int,
                        help='Number of RSFC samples per subject to run')
    return parser.parse_args()


def main():
    args = parse_test_arguments()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load subjects
    subj_ids = np.loadtxt(os.path.join(args.root, args.subj_list), dtype=str)

    # Instantiate model (adapt in_ch/out_ch to match your training)
    model = RPTR(mesh_dir='data/fs_LR_mesh_templates', in_ch=100, out_ch=2)
    model.to(device)

    # Load checkpoint
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['state_dict'])
    model.eval()

    # Inference
    with torch.no_grad():
        for i, subj in enumerate(subj_ids):
            print(f"Subject {i+1}/{len(subj_ids)}: {subj}")
            out_path = os.path.join(args.output_dir, f"{subj}_pred.npy")

            if not os.path.exists(out_path):
                # If you want multiple samples, loop over them
                preds = []
                for sample_id in range(args.num_samples):
                    rsfc_file = os.path.join(args.root, args.rsfc_dir,
                        f"joint_LR_{subj}_sample{sample_id}_rsfc.npy")
                    
                    # Load and prepare data
                    rsfc_data = np.expand_dims(np.load(rsfc_file), axis=0) # => [1, in_ch, #vertices]
                    rsfc_tensor = torch.from_numpy(rsfc_data).float().to(device)
                    
                    # Forward pass
                    sample_pred = model(rsfc_tensor)    # => [1, out_ch, #vertices]
                    sample_pred_np = sample_pred.cpu().numpy().squeeze(0)
                    preds.append(sample_pred_np)

                # Stack along sample dimension if needed
                preds = np.array(preds)  # shape: [num_samples, out_ch, #vertices]
                np.save(out_path, preds)
                print(f"Saved: {out_path}")

    print("Finished prediction.")


if __name__ == "__main__":
    main()
