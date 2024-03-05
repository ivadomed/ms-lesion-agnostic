"""
This script is used to evaluate the lesion segmentation predictions.
It iterates over the entire cohort.
It looks at different metrics (dice score, precision and recall) and saves the results in a csv file.
We use anima for the computation of the metrics. Here is the return message from animaSegPerfAnalyzer:
------------------------------------------------------------------------------------------------------------------------
3 categories are available:
    - SEGMENTATION EVALUATION:
        Dice, the mean overlap
        Jaccard, the union overlap
        Sensitivity
        Specificity
        NPV (Negative Predictive Value)
        PPV (Positive Predictive Value)
        RVE (Relative Volume Error) in percentage
    - SURFACE DISTANCE EVALUATION:
        Hausdorff distance
        Contour mean distance
        Average surface distance
    - DETECTION LESIONS EVALUATION:
        PPVL (Positive Predictive Value for Lesions)
        SensL, Lesion detection sensitivity
        F1 Score, a F1 Score between PPVL and SensL

Results are provided as follows: 
Jaccard;        Dice;   Sensitivity;    Specificity;    PPV;    NPV;    RelativeVolumeError;    HausdorffDistance;      ContourMeanDistance;    SurfaceDistance;        PPVL;   SensL;  F1_score;       NbTestedLesions;        VolTestedLesions;
------------------------------------------------------------------------------------------------------------------------

Args:
    --pred-folder, -p: BIDS format folder containing the predictions
    --dataset, -d: BIDS format containing the original dataset
    --animaPath, -a : path to the animaSegPerfAnalyzer script
    --output_folder, -o : path to the output folder

Returns:
    None

Example:
    python evaluate_lesion_seg_prediction.py --pred-folder path/to/predictions --mask-gt path/to/dataset --animaPath path/to/animaSegPerfAnalyzer --output-folder path/to/output_folder

TODO: 
    *

Pierre-Louis Benveniste
"""

import os
import pathlib
import sys
import argparse
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def get_parser():
    """
    This function parses the command line arguments and returns an argparse object.

    Input:
        None

    Returns:
        parser : argparse object
    """
    parser = argparse.ArgumentParser(description='Evaluate lesion segmentation prediction')
    parser.add_argument('--pred-folder', '-p', type=str, required=True, help='BIDS format folder containing the predictions')
    parser.add_argument('--mask-gt', '-d', type=str, required=True, help='BIDS format containing the mask GT')
    parser.add_argument('--animaPath', '-a', type=str, required=True, help='path to Anima scripts')
    parser.add_argument('--output-folder', '-o', type=str, required=True, help='path to the output folder')
    args = parser.parse_args()
    return args


def analyze_prediction(prediction, lesion_mask, animaPath, output_folder):
    """
    This script analyzes each prediction using animaSegPerfAnalyzer.
    Its created a csv file containing the results saved in the BIDS folder containing the predictions.

    Args:
        prediction : path to the prediction
        lesion_mask : path to the lesion mask
        animaPath : path to the animaSegPerfAnalyzer script
        output_folder : path to the output folder
    
    Returns:
        subject_results : a dictionnary containing the results of the analysis
    """

    #we initialize the results dictionnary
    subject_results = {}

    # we get the name of the subject
    name = pathlib.Path(prediction).name.replace('.nii.gz','')
    subject_results['name'] = name

    #we compute the metrics using animaSegPerfAnalyzer
    print("-------------------")
    print(f'Computing metrics for {name}' )
    print(f'Prediction: {prediction}')
    print(f'Lesion mask: {lesion_mask}')
    print("-------------------")
    result_output_path = os.path.join(output_folder, 'tmp',name)
    os.system( f'{animaPath}/animaSegPerfAnalyzer -r {lesion_mask} -i {prediction} -o {result_output_path} -d -s -l -X -S')
    print('Done!')

    #we read the xml file (such as this : singleClassNnunetMsLesion_278_metrics_global.xml)
    xml_file = result_output_path + '_global.xml'
    # check if file exists : if it doesn't exist we skip the subject
    if not os.path.isfile(xml_file):
        print(f'Error: {xml_file} does not exist. Skipping subject {name}')
        return None
    root_node = ET.parse(source=xml_file).getroot()
    for metric in root_node:
        metric_name, value = metric.get('name'), float(metric.text)
        subject_results[metric_name] = value
    
    return subject_results


def main():
    """
    This function is the main function of the script.

    Args:
        None
    
    Returns:
        None
    """
    #we parse the command line arguments
    args = get_parser()
    pred_folder = args.pred_folder
    mask_gt = args.mask_gt
    animaPath = args.animaPath
    output_folder = args.output_folder

    #we build the output folder if it does not exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if not os.path.isdir(os.path.join(output_folder, 'tmp')):
        os.makedirs(os.path.join(output_folder, 'tmp'))

    #we initialize the results dataframe
    results_dataframe = pd.DataFrame()

    #we get the list of predictions 
    predictions = list(pathlib.Path(pred_folder).rglob(f'*.nii.gz'))

    #we get the list of the lesion masks
    lesion_masks = list(pathlib.Path(mask_gt).rglob(f'*.nii.gz'))

    #we iterate over the predictions
    for prediction in predictions:
        
        #we get the name of the corresponding subject
        pred_name = pathlib.Path(prediction).name

        # fing the corresponding lesion mask
        lesion_mask = [mask for mask in lesion_masks if pred_name in str(mask)][0]

        #we analyze the prediction
        subject_results = analyze_prediction(prediction, lesion_mask, animaPath, output_folder)

        #we add the results to the dataframe
        results_dataframe = pd.concat([results_dataframe, pd.DataFrame([subject_results])], ignore_index=True)

    #we save the results dataframe in the output folder
    results_dataframe.to_csv(os.path.join(output_folder, 'results.csv'), index=False)
    
    return None


if __name__ == "__main__":
    main()
