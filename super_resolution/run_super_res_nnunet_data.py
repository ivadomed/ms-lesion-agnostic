"""
This algorithm runs the super-resolution model on some preprocessed data.
The nnUNet data is either some train or test data in the nnUNet format.
We give as input a min and max index to be a able to run the super-resolution in parallel.
This script requires to activate the smorve conda environment.

Input:
- input_dir: The directory containing the nnUNet data.
- output_dir: The directory where the super-resolution results will be saved.
- min_index: The minimum index of the data to process.
- max_index: The maximum index of the data to process.

Output:
    None

Example:
    python run_super_res_nnunet_data.py --input_dir /path/to/nnUNet_data --output_dir /path/to/output --min_index 0 --max_index 100

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import subprocess
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description="Run super-resolution on nnUNet data.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing nnUNet data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for super-resolution results.")
    parser.add_argument("--min-index", type=int, required=True, help="Minimum index of the data to process.")
    parser.add_argument("--max-index", type=int, required=True, help="Maximum index of the data to process.")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Build the output_dir if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # List all .nii.gz files in the input directory
    input_files = list(Path(input_dir).rglob("*.nii.gz"))
    input_files = sorted(input_files)

    # Images will have the following standard: name_XXX_0000.nii.gz with XXX being the index
    # We will process files from min_index to max_index
    input_files = [f for f in input_files if int(f.stem.split("_")[-2]) >= args.min_index and int(f.stem.split("_")[-2]) <= args.max_index]

    # Run the super-resolution model on each file
    for input_file in tqdm(input_files):
        # If an image is isotropic we just copy it to the destination folder
        img = nib.load(str(input_file))
        resolution = img.header.get_zooms()[:3]  # Get the first three dimensions (x, y, z)
        # If all three are clode to 1e-2 then we consider it isotropic
        if max(resolution) - min(resolution) < 1e-2:
            # Then we just copy the file to the output directory
             assert os.system(f"cp {str(input_file)} {output_dir}")==0
        else:
            # Build a temp folder
            temp_dir = Path(output_dir) / "temp"
            os.makedirs(temp_dir, exist_ok=True)
            os.system(f"run-smore --in-fpath {str(input_file)} --out-dir {temp_dir}")
            # Then we copy the result to the output directory
            temp_files = list(temp_dir.rglob("*.nii.gz"))
            assert os.system(f"cp {str(temp_files[0])} {str(Path(output_dir) / temp_files[0].name.replace('_smore4', ''))}") == 0
            # Remove the temp directory
            assert os.system(f"rm -rf {temp_dir}") == 0

        # # We want to catch this error from the SMORE code:  image is "isotropic" and cannot be run through SMORE
        # result = subprocess.run(
        #         ["run-smore", "--in-fpath", str(input_file), "--out-dir", str(temp_dir)],
        #         check=True,
        #         capture_output=True,
        #         text=True
        # )
        # # Check if "isotropic" and cannot be run through SMORE is in the captured text
        # if 'image is "isotropic" and cannot be run through SMORE' in result.stderr:
        #     print(f"SMORE failed for file {str(input_file).split('/')[-1]} because it was isotropic. Just copying the original file to the output directory.")
        #     assert os.system(f"cp {str(input_file)} {output_dir}")==0
        # else:
        #     print(result.stderr)

        # except subprocess.CalledProcessError as e:
        #     stderr = e.stderr.strip()
        #     print("fail")
        #     if "isotropic" in stderr.lower():
        #         print(f"Skipping {input_file}: isotropic image not supported by SMORE.")
        #     else:
        #         print(f"Error processing {input_file}:\n{stderr}")




        # # If SMORE crashed then temp folder will be empty
        # temp_files = list(temp_dir.rglob("*.nii.gz"))
        # if len(temp_files) == 0:
        #     print(f"SMORE failed for file {input_file} because it was isotropic. Just copying the original file to the output directory.")
        #     assert os.system(f"cp {str(input_file)} {output_dir}")==0
        #     continue
        # else: 
        #     assert os.system(f"cp {str(temp_files[0])} {str(Path(output_dir) / temp_files[0].name.replace('_smore4', ''))}") == 0
        # # Remove the temp directory
        # assert os.system(f"rm -rf {temp_dir}")==0

    return None


if __name__ == "__main__":
    main()