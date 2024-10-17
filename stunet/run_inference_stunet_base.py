"""
Inference of 3D STU-Net base model

This python script runs inference of the 3D STU-Net base model on individual nifti images.  

Example of run:
        $ python run_inference_stunet_base.py --path-image /path/to/image --path-out /path/to/output --path-model /path/to/model 

Arguments:
    --path-image : Path to the individual image to segment.
    --path-out : Path to output directory
    --path-model : Path to the model directory. This folder should contain individual folders like fold_0, fold_1, etc.'

Pierre-Louis Benveniste
"""
import os
import shutil
import argparse
import logging

import torch
import glob
import time

from image import Image, change_orientation, get_orientation, add_suffix
import numpy as np

# We define the environment variables here to avoid a warning from nnunetv2
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results']="./nnUNet_results"

# Import for nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

logger = logging.getLogger(__name__)


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--path-image', default=None,type=str)
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--path-model', required=True, 
                        help='Path to the model directory. This folder should contain individual folders '
                        'like fold_0, fold_1, etc.',)
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    return parser


def create_nnunet_from_plans(path_model, device: 'torch.device'):
    tile_step_size = 0.5
    # get the STunet trainer directory
    trainer_dirs = glob.glob(os.path.join(path_model, "STUNetTrainer*"))
    if len(trainer_dirs) != 1:
        raise FileNotFoundError(f"Could not find 'STUNetTrainer*' directory inside model path: {path_model} "
                                "Please make sure the release keeps the nnUNet output structure intact "
                                "by also including the 'nnUNetTrainer*' directory.")
    path_model = trainer_dirs[0]
    fold_dirs = [os.path.basename(path) for path in glob.glob(os.path.join(path_model, "fold_*"))]
    if not fold_dirs:
        raise FileNotFoundError(f"No 'fold_*' directories found in model path: {path_model}")
    folds_avail = 'all' if fold_dirs == ['fold_all'] else [int(f.split('_')[-1]) for f in fold_dirs]

    # We prioritize 'checkpoint_final.pth', but fallback to 'checkpoint_best.pth' if not available
    checkpoints = {os.path.basename(path) for path in glob.glob(os.path.join(path_model, "**", "checkpoint_*.pth"))}
    for checkpoint_name in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        if checkpoint_name in checkpoints:
            break  # Use the checkpoint that was found
    else:
        raise ValueError(f"Couldn't find 'checkpoint_final.pth' or 'checkpoint_best.pth' in {path_model}")
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,  # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,  # applies gaussian noise and gaussian blur
        use_mirroring=False,  # test time augmentation by mirroring on all axes
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print(f'Running inference on device: {predictor.device}')

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(path_model),
        use_folds=folds_avail,
        checkpoint_name=checkpoint_name,
    )
    print('Model loaded successfully. Fetching test data...')

    return predictor


def main():

    parser = get_parser()
    args = parser.parse_args()
    path_img = args.path_image
    path_out = args.path_out
    path_model = args.path_model

    # Create the output folder if it does not exist
    os.makedirs(path_out, exist_ok=True)
    tmpdir = os.path.join(path_out, 'tmp')
    os.makedirs(tmpdir, exist_ok=True)

    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # Get the original orientation of the image, for example LPI
    orig_orientation = get_orientation(Image(path_img_tmp))

    # Get the orientation used by the model
    model_orientation = "RPI"

    # Reorient the image to model orientation if not already
    img_in = Image(path_img_tmp)
    if orig_orientation != model_orientation:
        logger.info(f'Changing orientation of the input to the model orientation ({model_orientation})...')
        img_in.change_orientation(model_orientation)

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(path_img_tmp, "_pred")))
    os.makedirs(tmpdir_nnunet, exist_ok=True)

    # Run nnUNet prediction
    print('Starting inference...')
    start = time.time()

    # NOTE: nnUNet loads `.nii.gz` images using SimpleITK. When working with SimpleITK images, the axes are [x,y,z].
    #       But, during training, when nnUNet fetches a numpy array from the SimpleITK image, the axes get swapped
    #       ([z,y,x]). Nibabel (the image processing library SCT uses internally) _doesn't_ have this axis-swapping
    #       behavior. So, when SCT fetches the numpy array, we have to swap axes [x] and [z] to mimic nnUNet's internal
    #       behavior. See also: https://github.com/MIC-DKFZ/nnUNet/issues/1951
    data = img_in.data.transpose([2, 1, 0])
    # We also need to add an axis and convert to float32 to match nnUNet's input expectations
    # (This would automatically be done by nnUNet if we were to predict from image files, rather than a npy array.)
    data = np.expand_dims(data, axis=0).astype(np.float32)

    # Create the nnUNet predictor
    predictor = create_nnunet_from_plans(path_model, torch.device('cuda') if args.use_gpu else torch.device('cpu'))


    pred = predictor.predict_single_npy_array(
        input_image=data,
        # The spacings also have to be reversed to match nnUNet's conventions.
        image_properties={'spacing': img_in.dim[6:3:-1]},
    )
    # Lastly, we undo the transpose to return the image from [z,y,x] (SimpleITK) to [x,y,z] (nibabel)
    pred = pred.transpose([2, 1, 0])
    img_out = img_in.copy()
    img_out.data = pred

    end = time.time()
    print('Inference done.')
    total_time = end - start
    print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

    logger.info('Reorienting the prediction back to original orientation...')
    # Reorient the image back to original orientation
    if orig_orientation != model_orientation:
        img_out.change_orientation(orig_orientation)
        logger.info(f'Reorientation to original orientation {orig_orientation} done.')

    # Output path
    fname_prediction_final = os.path.join(path_out, os.path.basename(fname_prediction))
    print(f"Saving results to: {fname_prediction_final}")
    img_out.save(fname_prediction_final)

    # Remove temporary directories
    shutil.rmtree(tmpdir)
    
    return None


if __name__ == '__main__':
    main()