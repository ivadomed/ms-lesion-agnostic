"""
This script exports a trained nnUNet model with PyTorch to a file as well as the plans.json file which contains the model architecture.
WARNING: This script is only compatible with the environment which was used to train the model, i.e. it needs to have the trainer used for training (whihc can be custom).

Input: 
    -model-path: path to the trained nnUNet model folder
    -output-path: path to the output folder where the model will be saved
    -fold: fold number of the model to be exported (or 'foldall')

Output:
    None

Example:
    python export_nnunet_model.py --model-path /path/to/model/folders --output-path /output/path --fold 0

Author: Pierre-Louis Benveniste
"""

import os
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Export nnUNet model')
    parser.add_argument('--model-path', type=str, help='path to the trained nnUNet model folder')
    parser.add_argument('--output-path', type=str, help='path to the output folder where the model will be saved')
    parser.add_argument('--fold', type=str, help='fold number of the model to be exported (or "foldall")')
    return parser.parse_args()


def main():
    args = parse_args()
    path_to_model = args.model_path
    output_path = args.output_path
    fold = args.fold

    # Build the output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        path_to_model,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    model = predictor.network
    param = predictor.list_of_parameters[0]
    model.load_state_dict(param)

    # save the model
    torch.save(model.state_dict(), output_path + '/model.pth')

    # copy the plans.json file to the output folder
    shutil.copy(path_to_model + '/plans.json', output_path + '/plans.json')

    return None


if __name__ == "__main__":
    main()