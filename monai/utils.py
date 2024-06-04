import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch

import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

from scipy import ndimage



def dice_score(prediction, groundtruth, smooth=1.):
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
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    gt = gt.float().numpy()
    pred = pred.float().numpy()


    mid_sagittal = image.shape[0]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(3, 6, figsize=(10, 6))
    fig.suptitle('Original Image --> Ground Truth --> Prediction')
    for i in range(6):
        axs[0, i].imshow(image[mid_sagittal-3+i,:,:].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gt[mid_sagittal-3+i,:,:].T); axs[1, i].axis('off')
        axs[2, i].imshow(pred[mid_sagittal-3+i,:,:].T); axs[2, i].axis('off')

    # fig, axs = plt.subplots(1, 3, figsize=(10, 8))
    # fig.suptitle('Original Image --> Ground Truth --> Prediction')
    # slice = image.shape[2]//2

    # axs[0].imshow(image[:, :, slice].T, cmap='gray'); axs[0].axis('off') 
    # axs[1].imshow(gt[:, :, slice].T); axs[1].axis('off')
    # axs[2].imshow(pred[:, :, slice].T); axs[2].axis('off')
    
    plt.tight_layout()
    fig.show()
    return fig


def lesion_wise_tp_fp_fn(truth, prediction):
    """
    Computes the true positives, false positives, and false negatives two masks. Masks are considered true positives
    if at least one voxel overlaps between the truth and the prediction.
    Adapted from: https://github.com/npnl/atlas2_grand_challenge/blob/main/isles/scoring.py#L341

    Parameters
    ----------
    truth : array-like, bool
        3D array. If not boolean, will be converted.
    prediction : array-like, bool
        3D array with a shape matching 'truth'. If not boolean, will be converted.
    empty_value : scalar, float
        Optional. Value to which to default if there are no labels. Default: 1.0.

    Returns
    -------
    tp (int): 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp (int): 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn (int): 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    tp, fp, fn = 0, 0, 0

    # For each true lesion, check if there is at least one overlapping voxel. This determines true positives and
    # false negatives (unpredicted lesions)
    labeled_ground_truth, num_lesions = ndimage.label(truth.astype(bool))
    for idx_lesion in range(1, num_lesions+1):
        lesion = labeled_ground_truth == idx_lesion
        lesion_pred_sum = lesion + prediction
        if(np.max(lesion_pred_sum) > 1):
            tp += 1
        else:
            fn += 1

    # For each predicted lesion, check if there is at least one overlapping voxel in the ground truth.
    labaled_prediction, num_pred_lesions = ndimage.label(prediction.astype(bool))
    for idx_lesion in range(1, num_pred_lesions+1):
        lesion = labaled_prediction == idx_lesion
        lesion_pred_sum = lesion + truth
        if(np.max(lesion_pred_sum) <= 1):  # No overlap
            fp += 1

    return tp, fp, fn


def lesion_sensitivity(truth, prediction):
    """
    Computes the lesion-wise sensitivity between two masks
    Returns
    -------
    sensitivity (float): Lesion-wise sensitivity as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    empty_value = 1.0   # Value to which to default if there are no labels. Default: 1.0.

    if np.sum(truth) == 0 and np.sum(prediction)==0:
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    # if the prediction is not empty and ref is empty, it's false positive
    # if both are not empty, it's true positive
    else:

        tp, _, fn = lesion_wise_tp_fp_fn(truth, prediction)
        sensitivity = empty_value

        # Compute sensitivity
        denom = tp + fn
        if(denom != 0):
            sensitivity = tp / denom
        return sensitivity
        

def lesion_ppv(truth, prediction):
    """
    Computes the lesion-wise positive predictive value (PPV) between two masks
    Returns
    -------
    ppv (float): Lesion-wise positive predictive value as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    if np.sum(truth) == 0 and np.sum(prediction)==0:
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.sum(truth) != 0 and np.sum(prediction)==0:
        # Reference is not empty, prediction is empty --> model did not learn correctly (it's false negative)
        return 0.0
    # if the predction is not empty and ref is empty, it's false positive
    # if both are not empty, it's true positive
    else:
        tp, fp, _ = lesion_wise_tp_fp_fn(truth, prediction)
        # ppv = 1.0

        # Compute ppv
        denom = tp + fp
        # denom should ideally not be zero inside this else as it should be caught by the empty checks above
        if(denom != 0):
            ppv = tp / denom
        return ppv
    

def lesion_f1_score(truth, prediction):
    """
    Computes the lesion-wise F1-score between two masks by defining true positive lesions (tp), false positive lesions (fp)
    and false negative lesions (fn) using 3D connected-component-analysis.

    Masks are considered true positives if at least one voxel overlaps between the truth and the prediction.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    empty_value = 1.0   # Value to which to default if there are no labels. Default: 1.0.

    if np.sum(truth) == 0 and np.sum(prediction)==0:
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.sum(truth) != 0 and np.sum(prediction)==0:
        # Reference is not empty, prediction is empty --> model did not learn correctly (it's false negative)
        return 0.0
    # if the predction is not empty and ref is empty, it's false positive
    # if both are not empty, it's true positive
    else:
        tp, fp, fn = lesion_wise_tp_fp_fn(truth, prediction)
        f1_score = empty_value

        # Compute f1_score
        denom = tp + (fp + fn)/2
        if(denom != 0):
            f1_score = tp / denom
        return f1_score


def lesion_wise_precision_recall(prediction, groundtruth, iou_threshold=0.1):
    """
    This function computes the lesion-wise precision and recall.

    Args:
        prediction: predicted segmentation mask
        groundtruth: ground truth segmentation mask
        iou_threshold: threshold for intersection over union (IoU) for a lesion to be considered as true positive


# def lesion_wise_precision_recall(prediction, groundtruth, iou_threshold=0.1):
#     """
#     This function computes the lesion-wise precision and recall.

#     Args:
#         prediction: predicted segmentation mask
#         groundtruth: ground truth segmentation mask
#         iou_threshold: threshold for intersection over union (IoU) for a lesion to be considered as true positive
#     Returns:
#         precision: lesion-wise precision
#         recall: lesion-wise recall
#     """
#     prediction_cpu = prediction#.detach().numpy()
#     groundtruth_cpu = groundtruth#.detach().numpy()

#     precision = []
#     recall = []
#     # print(prediction_cpu.shape)
#     for i in range(prediction_cpu.shape[0]):
#         # Compute connected components in the predicted and ground truth segmentation masks
#         if len(prediction_cpu.shape) == 4:
#             # print("iteration")
#             # binarize the prediction and ground truth
#             prediction_cpu[0] = prediction_cpu[0] > 0.2
#             groundtruth_cpu[0] = groundtruth_cpu[0] > 0.2
#             # compute connected components
#             pred_labels, num_components_pred = skimage.measure.label(prediction_cpu[0], connectivity=2, return_num=True)
#             gt_labels, num_components_gt = skimage.measure.label(groundtruth_cpu[0], connectivity=2, return_num=True)
#             # print('c', pred_labels.shape)
#             # print('d', gt_labels.shape)
#         if len(prediction_cpu.shape) == 5:
#             # binarize the prediction and ground truth
#             prediction_cpu[i][0] = prediction_cpu[i][0] > 0.2
#             groundtruth_cpu[i][0] = groundtruth_cpu[i][0] > 0.2
#             # compute connected components
#             pred_labels, num_components_pred  = skimage.measure.label(prediction_cpu[i][0], connectivity=2, return_num=True)
#             gt_labels, num_components_gt = skimage.measure.label(groundtruth_cpu[i][0], connectivity=2, return_num=True)
#             # print('e', pred_labels.shape)
#             # print('f', gt_labels.shape)
        
#         # If there are no connected components in the predicted or ground truth segmentation masks we return 0 and continue
#         if num_components_gt==0 or num_components_pred==0:
#             precision+= [0]
#             recall+= [0]
#             continue

#         # Compute the intersection over union (IoU) between each pair of connected components
#         iou_matrix = np.zeros((np.max(pred_labels), np.max(gt_labels)))
#         intersection_matrix = np.zeros((np.max(pred_labels), np.max(gt_labels)))
#         for i in range(np.max(pred_labels)):
#             for j in range(np.max(gt_labels)):
#                 # Compute the intersection
#                 intersection = np.sum((pred_labels == i + 1) * (gt_labels == j + 1))
#                 # Compute the union
#                 union = np.sum((pred_labels == i + 1)) + np.sum((gt_labels == j + 1)) - intersection
#                 # Compute the IoU
#                 iou_matrix[i, j] = intersection / union
#                 # if iou_matrix[i, j] > 0:
#                     # print("iou_matrix", iou_matrix[i, j])
#                 # Compute the intersection
#                 intersection_matrix[i, j] = intersection
        
#         # # Compute lesion-wise precision and recall
#         # true_positives = np.sum(np.max(iou_matrix, axis=1) > iou_threshold)
#         # false_positives = np.sum(np.max(iou_matrix, axis=0) <= iou_threshold)
#         # false_negatives = np.sum(np.max(iou_matrix, axis=1) <= iou_threshold)
#         # precision += [true_positives / (true_positives + false_positives)]
#         # recall+= [true_positives / (true_positives + false_negatives)]

#         # Compute lesion-wise precision and recall
#         true_positives = np.sum(np.max(intersection_matrix, axis=1) > iou_threshold)
#         false_positives = np.sum(np.max(intersection_matrix, axis=0) <= iou_threshold)
#         false_negatives = np.sum(np.max(intersection_matrix, axis=1) <= iou_threshold)
#         precision += [true_positives / (true_positives + false_positives)]
#         recall+= [true_positives / (true_positives + false_negatives)]


#     # Put it back in cuda
#     precision = torch.tensor(precision).cuda()
#     recall = torch.tensor(recall).cuda()

#     print("precision", precision)
#     print("recall", recall)
#     return precision, recall


# ############################################################################################################
# #                               NNUNet's Model
# ############################################################################################################
# nnunet_plans = {
#     "UNet_class_name": "PlainConvUNet",
#     "UNet_base_num_features": 32,
#     "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2, 2],
#     "n_conv_per_stage_decoder": [2, 2, 2, 2, 2, 2],
#     "pool_op_kernel_sizes": [
#         [1, 1, 1],
#         [1, 2, 2],
#         [1, 2, 2],
#         [2, 2, 2],
#         [2, 2, 2],
#         [1, 2, 2], 
#         [1, 2, 2]
#     ],
#     "conv_kernel_sizes": [
#         [1, 3, 3],
#         [1, 3, 3],
#         [3, 3, 3],
#         [3, 3, 3],
#         [3, 3, 3],
#         [3, 3, 3], 
#         [3, 3, 3]
#     ],
#     "unet_max_num_features": 320,
# }


# # ======================================================================================================
# #                               Utils for nnUNet's Model
# # ====================================================================================================
# class InitWeights_He(object):
#     def __init__(self, neg_slope=1e-2):
#         self.neg_slope = neg_slope

#     def __call__(self, module):
#         if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
#             module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
#             if module.bias is not None:
#                 module.bias = nn.init.constant_(module.bias, 0)


# # ======================================================================================================
# #                               Define the network based on plans json
# # ====================================================================================================
# def create_nnunet_from_plans(plans=nnunet_plans, num_input_channels=1, num_classes=1, deep_supervision: bool = False):
#     """
#     Adapted from nnUNet's source code: 
#     https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9

#     """
#     num_stages = len(plans["conv_kernel_sizes"])

#     dim = len(plans["conv_kernel_sizes"][0])
#     conv_op = convert_dim_to_conv_op(dim)

#     segmentation_network_class_name = plans["UNet_class_name"]
#     mapping = {
#         'PlainConvUNet': PlainConvUNet,
#         'ResidualEncoderUNet': ResidualEncoderUNet
#     }
#     kwargs = {
#         'PlainConvUNet': {
#             'conv_bias': True,
#             'norm_op': get_matching_instancenorm(conv_op),
#             'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
#             'dropout_op': None, 'dropout_op_kwargs': None,
#             'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
#         },
#         'ResidualEncoderUNet': {
#             'conv_bias': True,
#             'norm_op': get_matching_instancenorm(conv_op),
#             'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
#             'dropout_op': None, 'dropout_op_kwargs': None,
#             'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
#         }
#     }
#     assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
#                                                               'is non-standard (maybe your own?). Yo\'ll have to dive ' \
#                                                               'into either this ' \
#                                                               'function (get_network_from_plans) or ' \
#                                                               'the init of your nnUNetModule to accomodate that.'
#     network_class = mapping[segmentation_network_class_name]

#     conv_or_blocks_per_stage = {
#         'n_conv_per_stage'
#         if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["n_conv_per_stage_encoder"],
#         'n_conv_per_stage_decoder': plans["n_conv_per_stage_decoder"]
#     }
    
#     # network class name!!
#     model = network_class(
#         input_channels=num_input_channels,
#         n_stages=num_stages,
#         features_per_stage=[min(plans["UNet_base_num_features"] * 2 ** i, 
#                                 plans["unet_max_num_features"]) for i in range(num_stages)],
#         conv_op=conv_op,
#         kernel_sizes=plans["conv_kernel_sizes"],
#         strides=plans["pool_op_kernel_sizes"],
#         num_classes=num_classes,    
#         deep_supervision=deep_supervision,
#         **conv_or_blocks_per_stage,
#         **kwargs[segmentation_network_class_name]
#     )
#     model.apply(InitWeights_He(1e-2))
#     if network_class == ResidualEncoderUNet:
#         model.apply(init_last_bn_before_add_to_0)
    
#     return model