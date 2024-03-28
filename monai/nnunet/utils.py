import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch

def dice_score(prediction, groundtruth):
    smooth = 1.
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice

# Check if any label image patch is empty in the batch
def check_empty_patch(labels):
    for i, label in enumerate(labels):
        if torch.sum(label) == 0.0:
            # print(f"Empty label patch found at index {i}. Skipping training step ...")
            return None
    return labels  # If no empty patch is found, return the labels

# Function to multiply by -1
def multiply_by_negative_one(x):
    return x * -1

def plot_slices(image, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy
    image = image.numpy()
    gt = gt.numpy()
    pred = pred.numpy()


    mid_sagittal = image.shape[2]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(3, 6, figsize=(10, 6))
    fig.suptitle('Original Image --> Ground Truth --> Prediction')
    for i in range(6):
        axs[0, i].imshow(image[:, :, mid_sagittal-3+i].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gt[:, :, mid_sagittal-3+i].T); axs[1, i].axis('off')
        axs[2, i].imshow(pred[:, :, mid_sagittal-3+i].T); axs[2, i].axis('off')

    # fig, axs = plt.subplots(1, 3, figsize=(10, 8))
    # fig.suptitle('Original Image --> Ground Truth --> Prediction')
    # slice = image.shape[2]//2

    # axs[0].imshow(image[:, :, slice].T, cmap='gray'); axs[0].axis('off') 
    # axs[1].imshow(gt[:, :, slice].T); axs[1].axis('off')
    # axs[2].imshow(pred[:, :, slice].T); axs[2].axis('off')
    
    plt.tight_layout()
    fig.show()
    return fig