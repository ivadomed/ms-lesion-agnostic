"""
Make sure to have installed ray tune:
    pip install ray[tune]<=2.9.3"

Performs a hyperparameter search on a yolov8n model, using ray tune.

To track with wandb, make sure wandb is installed (pip install wandb), 
then log into wandb account by following these steps: 
https://docs.ultralytics.com/integrations/weights-biases/#configuring-weights-biases 
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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

    args = parser.parse_args()

    # It seems like tune currently only works with the smallest model (n):
    # https://github.com/ultralytics/ultralytics/issues/2265
    model = YOLO('yolov8n.pt')

    # TODO take a config file as input
    # these are the suggested intervals on the ultralytics website:
    # param_space={"lr0": tune.uniform(1e-5, 1e-1),
    #              "lrf": tune.uniform(0.01, 1.0),
    #              "degrees": tune.uniform(0, 20.0),
    #              "scale": tune.uniform(0.5, 0.9),
    #              "fliplr": tune.uniform(0.0, 1.0),
    #              "translate": tune.uniform(0.0, 0.9),
    #              "hsv_v": tune.uniform(0.0, 0.9),
    #              "box": tune.uniform(0.02, 0.2),
    #              "cls": tune.uniform(0.2, 4.0)}
        
    param_space={"box": tune.uniform(0.1, 20.0),
                  "cls": tune.uniform(0.01, 5.0)}

    # setting epochs to 40 leads to an error
    # the error is avoided with 100 epochs, as suggested here: https://github.com/ultralytics/ultralytics/issues/5874
    result_grid = model.tune(data=args.data, 
                             use_ray=True, 
                             space=param_space,
                             iterations=16, # Number of runs 
                             epochs=100,
                             device=0, 
                             name=args.name, 
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

    if result_grid.errors:
        print("One or more trials failed!")
    else:
        print("No errors!")

    for i, result in enumerate(result_grid):
        print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")


if __name__ == "__main__":
    _main()
    