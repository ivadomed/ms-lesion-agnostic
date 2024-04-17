"""
Make sure to have installed ray tune:
    pip install "ray[tune]<=2.9.3"

Performs a hyperparameter search on a yolov8n model, using ray tune.

To track with wandb, make sure wandb is installed (pip install wandb), 
then log into wandb account by following these steps: 
https://docs.ultralytics.com/integrations/weights-biases/#configuring-weights-biases 
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
from pathlib import Path
from ultralytics import YOLO
from ray import tune


def _main():
    parser = ArgumentParser(
    prog = 'yolo_hyperparameter_tune',
    description = 'Perform a hyperparameter search on yolov8n model',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        required = True,
                        type = Path,
                        help = 'Path to data yaml file')
    parser.add_argument('-n', '--name',
                        default=None,
                        type = str,
                        help = 'Run name')
    parser.add_argument('-p', '--params',
                        default="default_tune_params.json",
                        type = Path,
                        help = 'Path to parameter file. See default_tune_params.json for format.')
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

    # It seems like tune currently only works with the smallest model (n):
    # https://github.com/ultralytics/ultralytics/issues/2265
    model = YOLO('yolov8n.pt')
    
    # Get parameters from params file
    with open(args.params, 'r') as file:
        config = json.load(file)

    param_space = {}
    fixed_params = {}
    for key, value in config.items():
        if isinstance(value, list):  # If the value is a list, it is a tuning parameter
            param_space[key] = tune.uniform(value[0], value[1])
        else:  # Otherwise, it's a fixed value
            fixed_params[key] = value

    # setting epochs to 40 leads to an error
    # the error is avoided with 100 epochs, as suggested here: https://github.com/ultralytics/ultralytics/issues/5874
    result_grid = model.tune(data=args.data, 
                             use_ray=True, 
                             space=param_space,
                             epochs=100,
                             device=device, 
                             name=args.name,
                             **fixed_params)

    if result_grid.errors:
        print("One or more trials failed!")
    else:
        print("No errors!")

    for i, result in enumerate(result_grid):
        print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")


if __name__ == "__main__":
    _main()
    