"""
Main script for pre-processing
Calls sc_seg_from_list.py, make_yolo_dataset.py and modify_unlabeled_proportion.py

Generates a YOLO dataset from a list of scans
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import subprocess
import tempfile


def call_sc_seg_from_list(json_path:str|Path, database:str|Path):
    """
    Calls sc_seg_from_list.py
    """
    print("Getting spinal cord segmentation...")
    command = [
    "python",
    "sc_seg_from_list.py",
    "-j", str(json_path),
    "-d", str(database)
    ]
    subprocess.run(command, check=True)

def call_make_yolo_dataset(json_path:str|Path, 
                           database:str|Path, 
                           output_dir:str|Path):
    """
    Calls make_yolo_dataset.py
    """
    print("Converting to YOLO format...")
    command = [
    "python",
    "make_yolo_dataset.py",
    "-j", str(json_path),
    "-d", str(database),
    "-o", str(output_dir)
    ]
    subprocess.run(command, check=True)

def call_modify_unlabeled_proportion(input_path:str|Path, 
                                     output_path:str|Path,
                                     ratio: str|float):
    """
    Calls modify_unlabeled_proportion.py
    """
    print("Modifying unlabeled proportion...")
    command = [
    "python",
    "modify_unlabeled_proportion.py",
    "-i", str(input_path),
    "-o", str(output_path),
    "-r", str(ratio)
    ]
    subprocess.run(command, check=True)

def _main():
    parser = ArgumentParser(
    prog = 'complete_pre_process',
    description = 'Generates YOLO format dataset from a list of scans and a BIDS database.',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--json-list',
                        required = True,
                        type = Path,
                        help = 'path to json list of scans to process')
    parser.add_argument('-d', '--database',
                        required = True,
                        type = Path,
                        help = 'path to BIDS database (canproco)')
    parser.add_argument('-o', '--output-dir',
                        required = True,
                        type = Path,
                        help = 'Output directory for YOLO dataset')
    parser.add_argument('-r', '--ratio',
                        default = None,
                        type = float,
                        help = 'Proportion of dataset that should be unlabeled. '
                               'By default, the ratio is not modified and the whole dataset is kept.')

    args = parser.parse_args()

    # Make sure all necessary spinal cord segmentations are present
    call_sc_seg_from_list(args.json_list, args.database)

    if args.ratio:
        # If ratio needs to be modified, call make_yolo_dataset in a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            call_make_yolo_dataset(args.json_list, args.database, Path(tmpdir)/"yolo_dataset")
            call_modify_unlabeled_proportion(Path(tmpdir)/"yolo_dataset", args.output_dir, args.ratio)

    else:
        # Otherwise, save dataset to output_dir directly
        call_make_yolo_dataset(args.json_list, args.database, args.output_dir)
    
    print(f"Dataset was saved to {args.output_dir}")


if __name__ == "__main__":
    _main()
