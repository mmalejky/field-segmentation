from __future__ import print_function
import argparse
import os
import time, platform
from tqdm import tqdm

import cv2
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from .losses import *
from .model import DexiNed
from .utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result,count_parameters)
from .dexi_utils import get_mean_and_std



OUTPUT_DIR = "."
MEAN_PATH = "./mean.pt"
TRAIN_DATA_PATH = "./data/edges_opole_pure"

def main():
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)

    # loading pretrained model on BIPED
    model.load_state_dict(torch.load("./baseline-model.pt",
                                        map_location=device))

    if not os.path.exists(MEAN_PATH):
        mean_std = get_mean_and_std(TRAIN_DATA_PATH)
        torch.save(mean_std, MEAN_PATH)
        print("Calculated new mean and std.")
    else:
        mean_std = torch.load(MEAN_PATH)
        print("Mean and std loaded from file.")

    mean, std = mean_std
    print(f"Mean and Std is: {mean_std}")

    # remember to always set img_width and img_height to be
    # more or equal the size of input images
    # BEWARE that not all sizes work, due to internal model mechanics
    dataset_train = BipedDataset(   TRAIN_DATA_PATH,
                                    img_width=128,
                                    img_height=128,
                                    mean_bgr = mean.tolist(),
                                    )

    dataloader_train = DataLoader(dataset_train,
                                    batch_size=8,
                                    shuffle=True)

    criterion = hed_loss2
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001,
                           amsgrad=True)

    # Main training loop
    seed=1021
    progress = tqdm(range(20))
    for epoch in progress:
        model.train()
        loss_avg =[]
        for sample_batched in dataloader_train:
            images = sample_batched['images'].to(device)  # BxCxHxW
            labels = sample_batched['labels'].to(device)  # BxHxW
            preds_list = model(images)
            loss = sum([criterion(preds, labels) for preds in preds_list])  #HED loss, rcf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg.append(loss.item())
            
        avg_loss = np.array(loss_avg).mean()
        progress.set_description(f"lr: {optimizer.param_groups[0]['lr']}, loss: {round(avg_loss, 2)}")
 
    torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
               os.path.join(OUTPUT_DIR, 'finetuned-model.pt'))

if __name__ == '__main__':
    main()
