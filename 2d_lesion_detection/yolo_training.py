"""
Trains a new YOLO model
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from ultralytics import YOLO


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
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data=args.data, epochs=args.epochs, name=args.name, device = device, mosaic=0, degrees=20, hsv_s=0, hsv_h=0)


if __name__ == "__main__":
    _main()

