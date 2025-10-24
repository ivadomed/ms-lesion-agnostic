"""
This code compares lesion segmentations from two timepoints for a single subject.

Input:
    -i1 : path to the input image at timepoint 1
    -i2 : path to the input image at timepoint 2
    -o : path to the output folder where comparison results will be stored

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import date
import nibabel as nib
from scipy import ndimage
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    return parser.parse_args()


def segment_sc(input_image, output_sc_seg):
    """
    This function segments the spinal cord from the input image and saves the segmentation.

    Inputs:
        input_image : path to the input image
        output_sc_seg : path to save the spinal cord segmentation
    Outputs:
        None
    """
    # Placeholder implementation
    logger.info(f"Segmenting spinal cord in {input_image} and saving to {output_sc_seg}")

    # Use SCT with GPU:
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {input_image} -o {output_sc_seg}") == 0, "Spinal cord segmentation failed"
    
    return None


def segment_lesions(input_image, input_sc_seg, qc_folder, output_lesion_seg):
    """
    This function segments lesions from the input image and saves the segmentation.

    Inputs:
        input_image : path to the input image
        input_sc_seg : path to the spinal cord segmentation
        qc_folder : path to the quality control folder
        output_lesion_seg : path to save the lesion segmentation

    Outputs:
        None
    """
    # Placeholder implementation
    logger.info(f"Segmenting lesions in {input_image} and saving to {output_lesion_seg}")

    # Use SCT with GPU:
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_image} -o {output_lesion_seg} -qc {qc_folder} -qc-plane Sagittal -qc-seg {input_sc_seg}") == 0, "Lesion segmentation failed"
    
    return None


def get_centerline(input_sc_seg, output_centerline):
    """
    This function computes the centerline from the spinal cord segmentation.

    Inputs:
        input_sc_seg : path to the spinal cord segmentation
        output_centerline : path to save the centerline
    """
    # Placeholder implementation
    logger.info(f"Computing centerline from {input_sc_seg} and saving to {output_centerline}")

    # Use SCT with GPU:
    assert os.system(f"sct_get_centerline -i  {input_sc_seg} -method fitseg -o {output_centerline}") == 0, "Centerline computation failed"

    return None


def get_levels(input_image, output_levels):
    """
    This function computes the vertebral levels from the input image.

    Inputs:
        input_image : path to the input image
        output_levels : path to save the vertebral levels
    """
    # Placeholder implementation
    logger.info(f"Computing vertebral levels from {input_image} and saving to {output_levels}")

    # Build a temporary folder for the levels
    temp_folder = os.path.join(os.path.dirname(output_levels), "temp_levels")
    os.makedirs(temp_folder, exist_ok=True)
    temp_output = os.path.join(temp_folder, 'output.nii.gz')

    # Use SCT with GPU:
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg totalspineseg -i {input_image} -step1-only 1 -o {temp_output} ") == 0, "Vertebral levels computation failed"

    # Then we only copy the output levels files to the desired output folder
    level_output_file = os.path.join(temp_output.replace('.nii.gz', '_step1_levels.nii.gz'))
    assert os.system(f'cp {level_output_file} {output_levels}') == 0, "Failed to copy vertebral levels file"

    # Clean up temporary folder
    os.system(f'rm -rf {temp_folder}')

    return None


def analyze_lesions(lesion_seg, sc_seg, centerline, levels, output_labeled_lesion_seg):
    """
    This function analyzes the lesions given the lesion segmentation, spinal cord segmentation, centerline, and vertebral levels.

    Inputs:
        lesion_seg : path to the lesion segmentation
        sc_seg : path to the spinal cord segmentation
        centerline : path to the centerline
        levels : path to the vertebral levels
    
    Outputs:
        analysis_results : results of the lesion analysis
    """
    # Placeholder implementation
    logger.info(f"Analyzing lesions in {lesion_seg}")

    # Initialize the analysis results
    analysis_results = {}

    # We load the segmentation and look at each lesion (connected component)
    lesion_data = nib.load(lesion_seg).get_fdata()
    # Label the connected components
    lbl_data, num_lesion = ndimage.label(lesion_data)
    # Compute the center of mass for each lesion
    labels = [i+1 for i in range(num_lesion)]
    h = ndimage.center_of_mass(lesion_data, lbl_data, labels)
    # Store the results
    analysis_results['num_lesions'] = num_lesion
    analysis_results['lesions'] = {}
    for label, CoM in zip(labels, h):
        analysis_results['lesions'][f'{label}'] = {}
        analysis_results['lesions'][f'{label}']['center_of_mass'] = CoM

    # For each lesion, we compute its volume
    voxel_volume = np.prod(nib.load(lesion_seg).header.get_zooms())
    for label in labels:
        lesion_data = (lbl_data == label).astype(np.uint8)
        lesion_volume = np.sum(lesion_data) * voxel_volume  # in mm^3
        analysis_results['lesions'][f'{label}']['volume_mm3'] = lesion_volume

    # We save a labeled lesion segmentation
    labeled_lesion_img = nib.Nifti1Image(lbl_data, nib.load(lesion_seg).affine)
    nib.save(labeled_lesion_img, output_labeled_lesion_seg)

    return analysis_results


def interpolate_between_levels(centerline_data, labeled_centerline_data, level_coords_1, level_coords_2, SI_axis, superior_level, resolution):
    """
    Interpolate the labeled values between two vertebral levels along the centerline.
    """
    # We get all points between these two levels (i.e. between SI values of level 1 and level 2)
    between_coords = []
    for coord in np.argwhere(centerline_data == 1):
        SI_value = coord[SI_axis]
        if SI_value <= level_coords_1[SI_axis] and SI_value >= level_coords_2[SI_axis]:
            between_coords.append(coord)
    # Remove level coords from between coords
    between_coords = [coord for coord in between_coords if not np.array_equal(level_coords_1, coord) and not np.array_equal(level_coords_2, coord)]
    # We interpolate the values for these points
    ## If dist to level 1 is d1 and dist to level 2 is d2, then the value is (d2/(d1+d2))*level1 + (d1/(d1+d2))*level2
    for coord in between_coords:
        dist_1 = np.min(np.linalg.norm((level_coords_1 - coord) * resolution))
        dist_2 = np.min(np.linalg.norm((level_coords_2 - coord) * resolution))
        total_dist = dist_1 + dist_2
        weight = dist_1 / total_dist
        labeled_value = weight + superior_level
        labeled_centerline_data[tuple(coord)] = labeled_value
    return labeled_centerline_data


def label_centerline(centerline, levels, output_labeled_centerline):
    """
    This function labels the centerline so that it takes continuous values following the disc levels.
    For example, the closest point to C1 is labeled 1, the closest point to C2 is labeled 2, and so on.
    For intermediate points, the value is interpolated.

    Inputs:
        centerline : path to the centerline
        levels : path to the vertebral levels
        output_labeled_centerline : path to save the labeled centerline

    Outputs:
        None
    """
    # Placeholder implementation
    logger.info(f"Labeling centerline {centerline} using levels {levels} and saving to {output_labeled_centerline}")

    # Load centerline and levels data
    centerline_data = nib.load(centerline).get_fdata()
    levels_data = nib.load(levels).get_fdata()

    # We get the SI direction of the image
    orientation = nib.aff2axcodes(nib.load(centerline).affine)
    SI_axis = orientation.index('I') if 'I' in orientation else orientation.index('S')

    # Get the resolution
    resolution = nib.load(centerline).header.get_zooms()

    # Create a copy of the centerline data to store the labeled values
    labeled_centerline_data = np.zeros_like(centerline_data)

    # For each level, assign the corresponding label to the closest centerline point
    level_labels = np.unique(levels_data)
    level_labels = level_labels[level_labels != 0]  # Exclude background
    for level in level_labels:
        # Get the coordinates of the current level
        level_coords = np.argwhere(levels_data == level)
        if level_coords.size == 0:
            continue
        # For each coordinate in the level, find the closest centerline point
        for coord in level_coords:
            distances = np.linalg.norm((np.argwhere(centerline_data == 1) - coord) * resolution, axis=1)
            closest_idx = np.argmin(distances)
            closest_point = np.argwhere(centerline_data == 1)[closest_idx]
            # Assign the level label to the closest centerline point
            labeled_centerline_data[tuple(closest_point)] = level
    
    # Interpolate values along the centerline
    for level, i in enumerate([min(level_labels)-1, *level_labels]):
        # We first take car of the centerline points that are above the first level
        if i == 0:
            superior_level = level + 1e-3 # A small value because we don't want it to be 0 like the background
            level_coords_2 = np.argwhere(labeled_centerline_data == level + 1)[0]
            # We consider that the upper level is at the top of the centerline (max IS value)
            level_coords_1 = np.argwhere(centerline_data == 1)[np.argmax(np.argwhere(centerline_data == 1)[:, SI_axis])]
            # And we assign a value for the superior level
            labeled_centerline_data[tuple(level_coords_1)] = superior_level
        # Then we take care of the centerline points that are below the last level
        elif i == len(level_labels):
            superior_level = level
            level_coords_1 = np.argwhere(labeled_centerline_data == level)[0]
            # We consider that the lower level is at the bottom of the centerline (min IS value)
            level_coords_2 = np.argwhere(centerline_data == 1)[np.argmin(np.argwhere(centerline_data == 1)[:, SI_axis])]
        # in this case, the centerline points are between two levels
        else:
            # We get the coordinates of the two levels
            level_coords_1 = np.argwhere(labeled_centerline_data == level)[0]
            level_coords_2 = np.argwhere(labeled_centerline_data == level + 1)[0]
            superior_level = level
        # Then we interpolate the values between these two levels
        labeled_centerline_data = interpolate_between_levels(centerline_data, labeled_centerline_data, level_coords_1, level_coords_2, SI_axis, superior_level, resolution)
    
    # Save the labeled centerline
    labeled_centerline_img = nib.Nifti1Image(labeled_centerline_data, nib.load(centerline).affine)
    nib.save(labeled_centerline_img, output_labeled_centerline)

    return None


def compute_theta_angle(CoM, z_value, CoM_closest_centerline_point, lbl_centerline, levels, resolution):
    """
    This function computes the theta angle of a lesion center of mass relative to the spinal cord anatomy.
    The angle is computed as the angle from the centerline-centerspine plane: 0 degree is anterior, 90 right, 180 posterior, 270 left.
    The A-P direction is defined as the average direction of the 3 direction at 3 levels: above, at, and below the lesion z value.
    
    Inputs:
        CoM : center of mass coordinates
        z_value : centerline z value of the lesion
        lbl_centerline : labeled centerline data
        levels : vertebral levels
        resolution : image resolution

    Outputs:
        theta : angle from centerline-centerspine plane
    """
    # We first need to define the anterior-posterior direction as the centerline-centerspine plane
    ## We find the axis which first better the centerline and the centerspine points.
    lbl_centerline_data = nib.load(lbl_centerline).get_fdata()
    levels_data = nib.load(levels).get_fdata()
    centerline_coords = np.argwhere(lbl_centerline_data > 0)
    ## We find the 3 closest levels to the lesion z value: one above, one at, and one below
    at_level = int(round(z_value))
    above_level = at_level - 1 if at_level - 1 >= np.min(levels_data[levels_data > 0]) else at_level
    below_level = at_level + 1 if at_level + 1 <= np.max(levels_data[levels_data > 0]) else at_level
    ## For each level of the centerspine, we get the vector from centerline to centerspine
    vectors = []
    for level in [above_level, at_level, below_level]:
        # Get the coordinates of the current level
        level_coord = np.argwhere(levels_data == level)
        # We find the closest point in the centerline
        distances_centerline = np.linalg.norm((centerline_coords - level_coord) * resolution, axis=1)
        closest_centerline_idx = np.argmin(distances_centerline)
        closest_centerline_point = centerline_coords[closest_centerline_idx]
        # Compute the vector from centerline to centerspine
        vector = (level_coord - closest_centerline_point ) * resolution
        vectors.append(vector)
    ## We average the vectors to get a mean anterior-posterior direction
    mean_vector = np.mean(vectors, axis=0).flatten()

    # Now we can compute the theta angle of the lesion CoM
    direction_lesion = (np.array(CoM) - CoM_closest_centerline_point) * resolution
    ## Compute angle in degrees between mean_vector and direction_lesion
    theta = np.arctan2(np.linalg.norm(np.cross(mean_vector, direction_lesion)), np.dot(mean_vector, direction_lesion))
    theta = np.degrees(theta)

    return theta


def compute_lesion_location(lesion_analysis, lesion_seg, sc_seg, lbl_centerline, levels):
    """
    This function computes the lesion location relative to spinal cord anatomy.
    The center of mass coordinates in this new space used the following parameters:
        -z: along the spinal cord centerline: if z = 1.5 -> lesion is located exactly between level C1 and C2
        -r: radial distance from the centerline (in mm)
        -theta: angle from centerline-centerspine plane: 0 degree is anterior, 90 right, 180 posterior, 270 left

    Inputs:
        lesion_analysis : results of the lesion analysis
        lesion_seg : path to the lesion segmentation
        sc_seg : path to the spinal cord segmentation
        lbl_centerline : path to the labeled centerline
        levels : path to the vertebral levels

    Outputs:
        updated_lesion_analysis : updated results of the lesion analysis with location information
    """
    # Load the data necessary for the computation
    lesion_data = nib.load(lesion_seg).get_fdata()
    lbl_centerline_data = nib.load(lbl_centerline).get_fdata()
    levels_data = nib.load(levels).get_fdata()

    # Get the resolution
    resolution = nib.load(lesion_seg).header.get_zooms()

    # iterate over each lesion:
    for lesion_id, lesion_info in lesion_analysis['lesions'].items():
        CoM = tuple(map(float, lesion_info['center_of_mass']))

        # Compute the z value: the closest point on the centerline from the center of mass
        ## Get all centerline coordinates
        centerline_coords = np.argwhere(lbl_centerline_data > 0)
        ## Compute distances from CoM to all centerline voxels
        distances = np.linalg.norm((centerline_coords - CoM) * resolution, axis=1)
        ## Find the closest centerline voxel
        closest_idx = np.argmin(distances)
        closest_centerline_point = centerline_coords[closest_idx]
        z_value = lbl_centerline_data[tuple(closest_centerline_point)]
        lesion_analysis['lesions'][lesion_id]['centerline_z'] = z_value

        # Compute the distance from the centerline (r) and angle (theta)
        r_value = distances[closest_idx] # in mm
        lesion_analysis['lesions'][lesion_id]['radius_mm'] = r_value

        # To compute theta, we need to define the anterior-posterior direction
        theta = compute_theta_angle(CoM, z_value, closest_centerline_point, lbl_centerline, levels, resolution)
        lesion_analysis['lesions'][lesion_id]['theta'] = theta

    return lesion_analysis


def lesion_matching(lesion_analysis_1, lesion_analysis_2):
    """
    This function performs lesion matching between two timepoints based on lesion location.

    Inputs:
        lesion_analysis_1 : results of the lesion analysis at timepoint 1
        lesion_analysis_2 : results of the lesion analysis at timepoint 2

    Outputs:
        matched_lesions : dictionary containing matched lesions between the two timepoints
    """
    # Placeholder implementation
    logger.info("Performing lesion matching between timepoint 1 and timepoint 2")

    # We build the Hungarian matrix with distances between lesions based on location (z, r, theta)
    hungarian_matrix = np.zeros((len(lesion_analysis_1['lesions']), len(lesion_analysis_2['lesions'])))

    # Define weights (tune empirically)
    w_z = 25.0      # strong weight on z-axis
    w_disk = 1.0    # small weight on axial position as angles are not very reliable

    for i, lesion_1 in enumerate(lesion_analysis_1['lesions'].values()):
        for j, lesion_2 in enumerate(lesion_analysis_2['lesions'].values()):
            # Compute distance between lesion_1 and lesion_2 based on (z, r, theta)
            distance_disk = np.sqrt(lesion_1['radius_mm']**2 + lesion_2['radius_mm']**2 
                                    - 2 * lesion_1['radius_mm'] * lesion_2['radius_mm'] * np.cos(np.radians(lesion_1['theta'] - lesion_2['theta'])))
            z_dist = lesion_1['centerline_z'] - lesion_2['centerline_z']
            hungarian_matrix[i, j] = np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)

    # We perform the Hungarian algorithm to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(hungarian_matrix)

    # We compute metrics of the lesions matching
    matched_lesions = {}

    for i, j in zip(row_ind, col_ind):
        # we print the information of the matched lesions
        lesion_1_data = lesion_analysis_1['lesions'][str(i+1)]
        lesion_1_data['previous_lesion_id'] = str(i+1)
        lesion_2_data = lesion_analysis_2['lesions'][str(j+1)]
        lesion_2_data['previous_lesion_id'] = str(j+1)
        # We build the dictionary of matched lesions
        matched_lesions[f'lesion_{i+1}'] = {
            'timepoint_1': lesion_1_data,
            'timepoint_2': lesion_2_data,
            'distance': hungarian_matrix[i, j],
        }
    
    # Get number of already matched lesions
    nb_matched_lesion = len(matched_lesions)
    
    # If a lesion is not matched in either timepoint, we add it as unmatched
    for lesion in range(len(lesion_analysis_1['lesions'])):
        if lesion not in row_ind:
            lesion_1_data = lesion_analysis_1['lesions'][str(lesion+1)]
            lesion_1_data['previous_lesion_id'] = str(lesion+1)
            matched_lesions[f'lesion_{nb_matched_lesion+1}'] = {
                'timepoint_1': lesion_1_data,
                'timepoint_2': None,
            }
            nb_matched_lesion += 1
        if lesion not in col_ind:
            lesion_2_data = lesion_analysis_2['lesions'][str(lesion+1)]
            lesion_2_data['previous_lesion_id'] = str(lesion+1)
            matched_lesions[f'lesion_{nb_matched_lesion+1}'] = {
                'timepoint_1': None,
                'timepoint_2': lesion_2_data,
            }
            nb_matched_lesion += 1

    return matched_lesions


def correct_lesion_labels(matched_lesions, labeled_lesion_seg_1, labeled_lesion_seg_2):
    """
    This function corrects the lesion labels in the lesion segmentations based on the matched lesions.

    Inputs:
        matched_lesions : dictionary containing matched lesions between the two timepoints
        labeled_lesion_seg_1 : path to the labeled lesion segmentation at timepoint 1
        labeled_lesion_seg_2 : path to the labeled lesion segmentation at timepoint 2
    
    Outputs:
        None
    """
    # Load the labeled lesion segmentations
    lbl_lesion_data_1 = nib.load(labeled_lesion_seg_1).get_fdata()
    lbl_lesion_data_2 = nib.load(labeled_lesion_seg_2).get_fdata()
    # Create new label data arrays
    new_lbl_lesion_data_1 = np.zeros_like(lbl_lesion_data_1)
    new_lbl_lesion_data_2 = np.zeros_like(lbl_lesion_data_2)

    # For each lesion in lbl_lesion_data_1, we assign its new label based on matched_lesions
    for lesion_key, lesion_info in matched_lesions.items():
        if lesion_info['timepoint_1'] is not None:
            previous_lesion_id = int(lesion_info['timepoint_1']['previous_lesion_id'])
            new_lesion_id = int(lesion_key.split('_')[1])
            new_lbl_lesion_data_1[lbl_lesion_data_1 == previous_lesion_id] = new_lesion_id
        if lesion_info['timepoint_2'] is not None:
            previous_lesion_id = int(lesion_info['timepoint_2']['previous_lesion_id'])
            new_lesion_id = int(lesion_key.split('_')[1])
            new_lbl_lesion_data_2[lbl_lesion_data_2 == previous_lesion_id] = new_lesion_id

    # Save the new label data arrays
    lbl_lesion_img_1 = nib.Nifti1Image(new_lbl_lesion_data_1, nib.load(labeled_lesion_seg_1).affine)
    lbl_lesion_img_2 = nib.Nifti1Image(new_lbl_lesion_data_2, nib.load(labeled_lesion_seg_2).affine)
    nib.save(lbl_lesion_img_1, labeled_lesion_seg_1)
    nib.save(lbl_lesion_img_2, labeled_lesion_seg_2)
    
    return None


def generate_report(matched_lesions, output_folder):
    """
    This function generates a report of the matched lesions and saves it to the output folder.

    Inputs:
        matched_lesions : dictionary containing matched lesions between the two timepoints
        output_folder : path to the output folder where the report will be saved
    Outputs:
        None
    """
    # Build the report file path, a csv file in the output folder
    report_file = os.path.join(output_folder, 'lesion_mapping_report.csv')
    with open(report_file, 'w') as f:
        # Write the header
        f.write('Lesion_ID,Timepoint_1_Volume_mm3,Timepoint_2_Volume_mm3,Timepoint_1_Centerline_z,Timepoint_2_Centerline_z, Timepoint_1_Radius_mm,Timepoint_2_Radius_mm,Timepoint_1_Theta,Timepoint_2_Theta,Distance\n')
        # Write the data for each matched lesion
        for lesion_key, lesion_info in matched_lesions.items():
            lesion_id = lesion_key.split('_')[1]
            tp1 = lesion_info['timepoint_1']
            tp2 = lesion_info['timepoint_2']
            distance = lesion_info.get('distance', '')
            tp1_volume = tp1['volume_mm3'] if tp1 is not None else ''
            tp1_z = tp1['centerline_z'] if tp1 is not None else ''
            tp1_r = tp1['radius_mm'] if tp1 is not None else ''
            tp1_theta = tp1['theta'] if tp1 is not None else ''
            tp2_volume = tp2['volume_mm3'] if tp2 is not None else ''
            tp2_z = tp2['centerline_z'] if tp2 is not None else ''
            tp2_r = tp2['radius_mm'] if tp2 is not None else ''
            tp2_theta = tp2['theta'] if tp2 is not None else ''
            f.write(f'{lesion_id},{tp1_volume},{tp2_volume},{tp1_z},{tp2_z},{tp1_r},{tp2_r},{tp1_theta},{tp2_theta},{distance}\n')
    logger.info(f"Report generated at {report_file}")
    return None


def map_lesions(input_image1, input_image2, output_folder):
    """
    This function performs lesion mapping between two timepoints.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where comparison results will be stored

    Outputs:
        None
    """
    # Build output directory
    os.makedirs(output_folder, exist_ok=True)
    # Build temporary directory
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    # Build QC folder
    qc_folder = os.path.join(output_folder, "qc")
    os.makedirs(qc_folder, exist_ok=True)

    # Build logger
    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))

    # Copy both images to the output folder
    assert os.system(f'cp {input_image1} {output_folder}/') == 0, "Failed to copy image 1"
    assert os.system(f'cp {input_image2} {output_folder}/') == 0, "Failed to copy image 2"

    # Segment the lesions, the sc, the centerline and the discs at both timepoints
    image_1_name = Path(input_image1).name
    lesion_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion_seg.nii.gz'))
    sc_seg_1 = os.path.join(output_folder, image_1_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    centerline_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_centerline.nii.gz'))
    levels_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
    image_2_name = Path(input_image2).name
    lesion_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion_seg.nii.gz'))
    sc_seg_2 = os.path.join(output_folder, image_2_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    centerline_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_centerline.nii.gz'))
    levels_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))

    # Segment the spinal cord
    segment_sc(input_image1, sc_seg_1)
    segment_sc(input_image2, sc_seg_2)
    # Segment the lesions
    segment_lesions(input_image1, sc_seg_1, qc_folder, lesion_seg_1)
    segment_lesions(input_image2, sc_seg_2, qc_folder, lesion_seg_2)
    # Get the centerline
    get_centerline(sc_seg_1, centerline_1)
    get_centerline(sc_seg_2, centerline_2)
    # Get the levels
    get_levels(input_image1, levels_1)
    get_levels(input_image2, levels_2)

    # Now we can perform lesion mapping between timepoint 1 and timepoint 2
    logger.info("Performing lesion mapping between timepoint 1 and timepoint 2")

    # For each timepoint, we analyze lesions
    labeled_lesion_seg_1 = os.path.join(output_folder, Path(lesion_seg_1).name.replace('.nii.gz', '_labeled.nii.gz'))
    labeled_lesion_seg_2 = os.path.join(output_folder, Path(lesion_seg_2).name.replace('.nii.gz', '_labeled.nii.gz'))
    lesion_analysis_1 = analyze_lesions(lesion_seg_1, sc_seg_1, centerline_1, levels_1, labeled_lesion_seg_1)
    lesion_analysis_2 = analyze_lesions(lesion_seg_2, sc_seg_2, centerline_2, levels_2, labeled_lesion_seg_2)

    # We label the centerline so that it takes continous various following the disc levels
    labeled_centerline_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_labeled_centerline.nii.gz'))
    labeled_centerline_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_labeled_centerline.nii.gz'))
    label_centerline(centerline_1, levels_1, labeled_centerline_1)
    label_centerline(centerline_2, levels_2, labeled_centerline_2)

    # We compute lesion location relative to spinal cord anatomy
    lesion_analysis_1 = compute_lesion_location(lesion_analysis_1, lesion_seg_1, sc_seg_1, labeled_centerline_1, levels_1)
    lesion_analysis_2 = compute_lesion_location(lesion_analysis_2, lesion_seg_2, sc_seg_2, labeled_centerline_2, levels_2)

    # We perform lesion matching between timepoint 1 and timepoint 2 based on location
    matched_lesions = lesion_matching(lesion_analysis_1, lesion_analysis_2)

    # Need to a labelization of lesion segmentations
    correct_lesion_labels(matched_lesions, labeled_lesion_seg_1, labeled_lesion_seg_2)

    # Need to add report generation
    generate_report(matched_lesions, output_folder)

    return None


def main():
    args = parse_args()
    input_image1 = args.input_image1
    input_image2 = args.input_image2
    output_folder = args.output_folder

    map_lesions(input_image1, input_image2, output_folder)

    return None


if __name__ == "__main__":
    main()