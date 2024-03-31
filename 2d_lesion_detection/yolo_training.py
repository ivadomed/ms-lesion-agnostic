"""
Script to train a YOLOv8 model

Training progress can be tracked using clearML (other platforms are also integrated but haven't been tested):
https://docs.ultralytics.com/integrations/clearml/#configuring-clearml

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
from ultralytics import YOLO, settings

CANPROCO_VERSION = "bcd627ed4" # last commit from canproco repo

def _main():
    parser = ArgumentParser(
    prog = 'yolo_training',
    description = 'Train a YOLOv8 model',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        required = True,
                        type = Path,
                        help = 'Path to data yaml file')
    parser.add_argument('-e', '--epochs',
                        default= 10,
                        type = int,
                        help = 'Number of epochs')
    parser.add_argument('-n', '--name',
                        default= None,
                        type = str,
                        help = 'Model name')
    parser.add_argument('-g', '--device',
                        default= 0,
                        type = str,
                        help = 'Device to use. 0 refers to gpu 0')

    args = parser.parse_args()

    # Get device as int if input is int
    try:
        device = int(args.device)  # Try converting to an integer
    except ValueError:
        device = args.device  # keep as string

    # Load a pretrained model
    model = YOLO('yolov8m.pt')

    # Train the model
    model.train(data=args.data,
                epochs=args.epochs,
                name=args.name,
                device = device,
                box=15.6,
                cls=4.1,
                mosaic=0,
                hsv_s=0,
                hsv_h=0,
                lr0=0.09,
                lrf=0.08,
                degrees=10,
                scale=0.5,
                fliplr=0.25,
                translate=0.25,
                hsv_v=0.45)


    ## Add canproco version to model metadata
        # Assuming no project is given to model.train
        # If that becomes the case, settings["runs_dir"]/"detect" needs to be replaced by project

    # Define the metadata
    metadata = {'dataset_version': CANPROCO_VERSION,
                'description': 'Model for multiple sclerosis lesion detection on MRI images of spinal cord.'}
    save_dir = model.trainer.save_dir

    # best.py
    print("Adding metadata to best.py")
    model_path = save_dir/"weights"/"best.pt"
    best_model = torch.load(model_path)
    best_model["metadata"] = metadata
    torch.save(best_model, model_path)

    # last.py
    print("Adding metadata to last.py")
    model_path = save_dir/"weights"/"last.pt"
    last_model = torch.load(model_path)
    last_model["metadata"] = metadata
    torch.save(last_model, model_path)

    # # To access the metadata:
        # metadata_read = torch_model.get('metadata', None)


if __name__ == "__main__":
    _main()
