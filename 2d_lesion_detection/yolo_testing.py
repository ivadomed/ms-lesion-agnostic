"""
Evaluates the performance of a YOLO model on the 'test' dataset
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from ultralytics import YOLO


def _main():
    parser = ArgumentParser(
    prog = 'yolo_testing',
    description = 'Evaluate performance of a trained YOLO model on the test dataset',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--trained_model',
                        required = True,
                        type = Path,
                        help = 'Path to trained model .pt file')
    parser.add_argument('-d', '--data',
                        default= None,
                        type = Path,
                        help = 'Path to data .yaml file. If none is given, the test set from' 
                                'the yaml file used for model training will be used')
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

    # Load trained model
    model = YOLO(args.trained_model)

    # Validate on the test set
    # iou and conf are default values
    model.val(data=args.data, split='test', plots=True, device = device, save_json=True, iou=0.7, conf=0.001)

if __name__ == "__main__":
    _main()