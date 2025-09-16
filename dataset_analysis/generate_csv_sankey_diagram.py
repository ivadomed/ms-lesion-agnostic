"""
This file is used to generate the CSV dataset necessary for the creation of the Sankey diagram.

Input:
    -i: path to MSD dataset file
    -o: path to output csv file

Output:
    None

Example:
    python generate_csv_sankey_diagram.py -i /path/to/input.json -o /path/to/output.csv

Author: Pierre-Louis Benveniste
"""
import argparse
import json
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CSV for Sankey diagram")
    parser.add_argument("-i", "--input", required=True, help="Path to MSD dataset file")
    parser.add_argument("-o", "--output", required=True, help="Path to output CSV file")
    return parser.parse_args()


def main():

    # Parse arguments:
    args = parse_args()

    path_msd = args.input
    path_output_csv = args.output

    # If output doesn't exist, we create it
    os.makedirs(os.path.dirname(path_output_csv), exist_ok=True)

    # Load the JSON file
    with open(path_msd, 'r') as f:
        data = json.load(f)

    subjects = data['train'] + data['validation'] + data['test'] + data['externalValidation']

    # Create a panda df with columns site (numbered sites), acquisition and contrast
    df = pd.DataFrame()

    # For each element in the json file add the element to the df
    for i in subjects:
        site = i['site']
        # If the site is canproco, we split it into the 5 site corresponds sites
        if site=='canproco':
            site = 'canproco_'+i['image'].split('/')[-1][4:7]
        acquisition = i['acquisition']
        if acquisition == 'ax':
            acquisition = '2D axial'
        elif acquisition == 'sag':
            acquisition = '2D sagittal'
        contrast = i['contrast']
        if contrast == 'MEGRE':
            contrast = 'T2*w'
        
        # Add the element to the df
        df = pd.concat([df, pd.DataFrame({'site': [site], 'acquisition': [acquisition], 'contrast': [contrast]})], ignore_index=True)
    # Create a dictionary to map site names to numbers
    site_dict = {}
    for i, site in enumerate(df['site'].unique()):
        site_dict[site] = f"site {i+1}"
    # Replace site names with numbers
    df['site'] = df['site'].map(site_dict)
    # Save the df to a csv file
    df.to_csv(path_output_csv, index=False)


if __name__ == "__main__":
    main()