"""
This code selects some images which sould be reviewed by the neuro-radiologists.
It deals with the images from the test set and the external validation set separately.
The numbers of images per contrast are defined at the beginning of the script.
The images are selected randomly. They are anonymized and saved in a folder with a conversion dict.

Input: 
    -path_conversion_dict: Path to the conversion dict
    -path_to_images: Path to the images folder
    -path_to_labels: Path to the labels folder
    -path_to_preds: Path to the predictions folder
    -output_path: Path to the output folder
    -path_conversion_dict_ext: Path to the conversion dict of the external validation set
    -path_to_images_ext: Path to the images folder of the external validation set
    -path_to_labels_ext: Path to the labels folder of the external validation set
    -path_to_preds_ext: Path to the predictions folder of the external validation set

Output:
    None

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import shutil
import json
import numpy as np
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path-conversion-dict", required=True, type=str, help="Path to the conversion dict")
    parser.add_argument("-path-to-images", required=True, type=str, help="Path to the images folder")
    parser.add_argument("-path-to-labels", required=True, type=str, help="Path to the labels folder")
    parser.add_argument("-path-to-preds", required=True, type=str, help="Path to the predictions folder")
    parser.add_argument("-output-path", required=True, type=str, help="Path to the output folder")
    parser.add_argument("-path-conversion-dict-ext", required=True, type=str, help="Path to the conversion dict of the external validation set")
    parser.add_argument("-path-to-images-ext", required=True, type=str, help="Path to the images folder of the external validation set")
    parser.add_argument("-path-to-labels-ext", required=True, type=str, help="Path to the labels folder of the external validation set")
    parser.add_argument("-path-to-preds-ext", required=True, type=str, help="Path to the predictions folder of the external validation set")
    return parser.parse_args()


def main(): 
    # Parse arguments
    args = parse_args()
    path_conversion_dict = args.path_conversion_dict
    path_to_images = args.path_to_images
    path_to_labels = args.path_to_labels
    path_to_preds = args.path_to_preds
    output_path = args.output_path
    path_conversion_dict_ext = args.path_conversion_dict_ext
    path_to_images_ext = args.path_to_images_ext
    path_to_labels_ext = args.path_to_labels_ext
    path_to_preds_ext = args.path_to_preds_ext

    # Initiliaze the seed
    np.random.seed(40)

    # Create the output folders
    os.makedirs(output_path, exist_ok=True)
    image_output_path = os.path.join(output_path, 'images')
    os.makedirs(image_output_path, exist_ok=True)
    label_output_path = os.path.join(output_path, 'labels')
    os.makedirs(label_output_path, exist_ok=True)

    # Save a conversion dict with the original name and the anonymized name
    conversion_dict_output = {}

    # Number of images
    n_unit1 = 2
    n_t2 = 9
    n_stir = 2
    n_psir = 3
    n_t2star = 3
    n_t1 = 1

    # Load the conversion dict
    with open(path_conversion_dict, 'r') as f:
        conversion_dict = json.load(f)

    # In the conversion dict, keep only the images which are in the test set , i.e. the field contains imagesTs
    conversion_dict = {k: v for k, v in conversion_dict.items() if "imagesTs" in v}

    # SELECTION OF UNIT1 IMAGES
    unit1_images = {k: v for k, v in conversion_dict.items() if "_UNIT1." in k}
    keys = list(unit1_images.keys())
    np.random.shuffle(keys)
    # unit1_images = {k: unit1_images[k] for k in keys[:n_unit1]}


    # SELECTION OF T2 IMAGES
    t2_images = {k: v for k, v in conversion_dict.items() if "_T2w" in k}
    keys = list(t2_images.keys())
    np.random.shuffle(keys)
    # t2_images = {k: t2_images[k] for k in keys[:n_t2]}

    # SELECTION OF STIR IMAGES
    stir_images = {k: v for k, v in conversion_dict.items() if "_STIR." in k}
    keys = list(stir_images.keys())
    np.random.shuffle(keys)
    # stir_images = {k: stir_images[k] for k in keys[:n_stir]}

    # SELECTION OF PSIR IMAGES
    psir_images = {k: v for k, v in conversion_dict.items() if "_PSIR." in k}
    keys = list(psir_images.keys())
    np.random.shuffle(keys)
    # psir_images = {k: psir_images[k] for k in keys[:n_psir]}

    # SELECTION OF T2STAR IMAGES
    t2star_images = {k: v for k, v in conversion_dict.items() if "_T2star." in k}
    keys = list(t2star_images.keys())
    np.random.shuffle(keys)
    # t2star_images = {k: t2star_images[k] for k in keys[:n_t2star]}

    # Aggregate all the images
    images_to_be_reviewed = {**unit1_images, **t2_images, **stir_images, **psir_images, **t2star_images}

    # count the number of images per contrast
    cnt_unit1 = 0
    cnt_t2 = 0
    cnt_stir = 0
    cnt_psir = 0
    cnt_t2star = 0

    # iterate through the different lists of images
    for contrast_list in [unit1_images, t2_images, stir_images, psir_images, t2star_images]:
        for elem in contrast_list:
            # Get the image name and path
            image_name = elem.split("/")[-1]
            image_path = images_to_be_reviewed[elem]
            # Get label and prediction path
            label_file_name = images_to_be_reviewed[elem].split('/')[-1].replace('_0000', '')
            label_path = os.path.join(path_to_labels, label_file_name)
            pred_path = os.path.join(path_to_preds, label_file_name)
            # check if all the files exist
            if not os.path.exists(image_path) or not os.path.exists(label_path) or not os.path.exists(pred_path):
                print("One of the files does not exist")
                break
            # We check if both the label and the prediction are not empty segmentation
            label_data = nib.load(label_path).get_fdata()
            pred_data = nib.load(pred_path).get_fdata()
            if np.sum(label_data) == 0 and np.sum(pred_data) == 0:
                continue
            # We add the count depending on the contrast
            if "_UNIT1." in elem:
                cnt_unit1 += 1
            elif "_T2w" in elem:
                cnt_t2 += 1
            elif "_STIR." in elem:
                cnt_stir += 1
            elif "_PSIR." in elem:
                cnt_psir += 1
            elif "_T2star." in elem:
                cnt_t2star += 1

            # Initialize the conversion dict output element
            conversion_dict_output[elem] = {}
            # Copy the files to the output folder
            shutil.copy(image_path, os.path.join(image_output_path, image_name))
            # Save the conversion dict output element
            conversion_dict_output[elem]['image'] = image_name
            # For the labels they take image_labelA.nii.gz or image_labelB.nii.gz (with random choice)
            if np.random.rand() > 0.5:
                shutil.copy(label_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelA.nii.gz')))
                conversion_dict_output[elem]['label'] = image_name.replace('.nii.gz', '_labelA.nii.gz')
                shutil.copy(pred_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelB.nii.gz')))
                conversion_dict_output[elem]['pred'] = image_name.replace('.nii.gz', '_labelB.nii.gz')
            else:
                shutil.copy(label_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelB.nii.gz')))
                conversion_dict_output[elem]['label'] = image_name.replace('.nii.gz', '_labelB.nii.gz')
                shutil.copy(pred_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelA.nii.gz')))
                conversion_dict_output[elem]['pred'] = image_name.replace('.nii.gz', '_labelA.nii.gz')
            
            # If the maximum number of images per contrast is reached, we move to the next contrast
            if cnt_unit1 == n_unit1 and contrast_list == unit1_images:
                break
            if cnt_t2 == n_t2 and contrast_list == t2_images:
                break
            if cnt_stir == n_stir and contrast_list == stir_images:
                break
            if cnt_psir == n_psir and contrast_list == psir_images:
                break
            if cnt_t2star == n_t2star and contrast_list == t2star_images:
                break
            
    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    # SELECTION OF T1 IMAGES IN THE EXTERNAL VALIDATION SET

    # Load the conversion dict ext
    with open(path_conversion_dict_ext, 'r') as f:
        conversion_dict_ext = json.load(f)

    # In the conversion dict, keep only the images(not the labels) which are in imagesExternalTs
    conversion_dict_ext = {k: v for k, v in conversion_dict_ext.items() if "imagesExternalTs" in v}

    # Select the T1w images
    t1_images = {k: v for k, v in conversion_dict_ext.items() if "_T1w" in k}
    keys = list(t1_images.keys())
    np.random.shuffle(keys)
    # t1_images = {k: t1_images[k] for k in keys[:n_t1]}

    cnt_t1 = 0

    # iterate over the T1w images
    for elem in t1_images:
        
        # Get the image name and path
        image_name = elem.split("/")[-1]
        image_path = t1_images[elem]
        # Get label and prediction path
        label_file_name = t1_images[elem].split('/')[-1].replace('_0000', '')
        label_path = os.path.join(path_to_labels_ext, label_file_name)
        pred_path = os.path.join(path_to_preds_ext, label_file_name)
        # check if all the files exist
        if not os.path.exists(image_path) or not os.path.exists(label_path) or not os.path.exists(pred_path):
            print("One of the files does not exist")
            break

        #  We check if both the label and the prediction are not empty segmentation
        label_data = nib.load(label_path).get_fdata()
        pred_data = nib.load(pred_path).get_fdata()
        if np.sum(label_data) == 0 and np.sum(pred_data) == 0:
            continue

        # We add the count depending on the contrast
        cnt_t1 += 1

        # Initialize the conversion dict output element
        conversion_dict_output[elem] = {}
        # Copy the files to the output folder
        shutil.copy(image_path, os.path.join(image_output_path, image_name))
        # Save the conversion dict output element
        conversion_dict_output[elem]['image'] = image_name
        # For the labels they take image_labelA.nii.gz or image_labelB.nii.gz (with random choice)
        if np.random.rand() > 0.5:
            shutil.copy(label_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelA.nii.gz')))
            conversion_dict_output[elem]['label'] = image_name.replace('.nii.gz', '_labelA.nii.gz')
            shutil.copy(pred_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelB.nii.gz')))
            conversion_dict_output[elem]['pred'] = image_name.replace('.nii.gz', '_labelB.nii.gz')
        else:
            shutil.copy(label_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelB.nii.gz')))
            conversion_dict_output[elem]['label'] = image_name.replace('.nii.gz', '_labelB.nii.gz')
            shutil.copy(pred_path, os.path.join(label_output_path, image_name.replace('.nii.gz', '_labelA.nii.gz')))
            conversion_dict_output[elem]['pred'] = image_name.replace('.nii.gz', '_labelA.nii.gz')

        # If the maximum number of images per contrast is reached, we move to the next contrast
        if cnt_t1 == n_t1:
            break

    # Save the conversion dict output
    with open(os.path.join(output_path, 'conversion_dict.json'), 'w') as f:
        json.dump(conversion_dict_output, f, indent=4)


if __name__ == "__main__":
    main()