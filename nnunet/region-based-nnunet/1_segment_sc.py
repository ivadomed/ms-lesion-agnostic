"""
In this script we segment the spinal cord of all the files which have a lesion segmentation file but no spinal cord segmentation file.
This script also creates a QC report of the resulting segmentation files for visual inspection and manual correction if necessary.
The segmentations are performed using the Spinal Cord Toolbox (SCT) v6.2: sct_deepseg -task sec_sc_contrast_agnostic
"""