from scipy import ndimage
import numpy as np


def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


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

    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.any(truth) and not np.any(prediction):
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
    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.any(truth) and not np.any(prediction):
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

    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    # if the predction is not empty and ref is empty, it's false positive
    # if both are not empty, it's true positive
    else:

        tp, _, fn = lesion_wise_tp_fp_fn(truth, prediction)
        sensitivity = empty_value

        # Compute sensitivity
        denom = tp + fn
        if(denom != 0):
            sensitivity = tp / denom
        return sensitivity