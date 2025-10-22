import torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth

class nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4000

# using a standardized function name so that SCT can import the class
def get_trainer_class():
   return nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch