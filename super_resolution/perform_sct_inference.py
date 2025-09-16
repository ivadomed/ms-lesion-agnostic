"""
This code performs inference on the super-resolved images using the SCT lesion_ms model.
The model used is the model from this release: https://github.com/ivadomed/ms-lesion-agnostic/releases/tag/r20250909
Inference is performed with GPU to speed up the process.
Instruction to use GPU inference with SCT: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4360#issuecomment-2035294418

Input:
    -input: The directory containing the super-resolved images.
    -output: The directory to save the inference results.

Output:
    None

Example:
    python perform_sct_inference.py --input /path/to/superres_images --output /path/to/output

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Perform SCT inference on super-resolved images.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing super-resolved images.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for inference results.")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    input_dir = args.input
    output_dir = args.output

    # Build the output_dir if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # List all .nii.gz files in the input directory
    input_files = list(Path(input_dir).rglob("*.nii.gz"))
    input_files = sorted(input_files)
    input_files = [str(f) for f in input_files]

    # Run the SCT model on each file
    for input_file in tqdm(input_files):
        # Build a temp folder to store intermediate results
        temp_folder = Path(output_dir) / "temp"
        os.makedirs(temp_folder, exist_ok=True)
        
        # Build file names
        output_temp_file = os.path.join(temp_folder, str(input_file).split('/')[-1].replace("_0000.nii.gz", ".nii.gz"))
        output_label_file = os.path.join(Path(output_dir), output_temp_file.split('/')[-1])

        # Run the inference command using GPU
        command = f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_file} -o {output_temp_file}"
        assert os.system(command) == 0

        # Move the resulting label file to the output directory
        assert os.path.exists(output_temp_file), f"Error: Output file {output_temp_file} not found."
        shutil.move(str(output_temp_file), str(output_label_file))

        # Remove the temporary folder
        shutil.rmtree(temp_folder)


if __name__ == "__main__":
    main()