
import numpy as np
from monai.transforms import (Compose, CropForegroundd, LoadImaged, RandFlipd, 
            Spacingd, RandScaleIntensityd, NormalizeIntensityd, RandAffined,
            DivisiblePadd, RandAdjustContrastd, EnsureChannelFirstd, RandGaussianNoised, 
            RandGaussianSmoothd, Orientationd, Rand3DElasticd, RandBiasFieldd, 
            RandSimulateLowResolutiond, ResizeWithPadOrCropd)


def train_transforms(crop_size, lbl_key="label"):

    monai_transforms = [    
        # pre-processing
        LoadImaged(keys=["image", lbl_key]),
        EnsureChannelFirstd(keys=["image", lbl_key]),
        # NOTE: spine interpolation with order=2 is spline, order=1 is linear
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
        ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
        # data-augmentation
        RandAffined(keys=["image", lbl_key], mode=(2, 1), prob=0.9,
                    rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians 
                    scale_range=(-0.2, 0.2),   
                    translate_range=(-0.1, 0.1)),
        Rand3DElasticd(keys=["image", lbl_key], prob=0.5,
                       sigma_range=(3.5, 5.5), 
                       magnitude_range=(25., 35.)),
        RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
        RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),    # this is monai's RandomGamma
        RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
        RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
        RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),  # this is nnUNet's BrightnessMultiplicativeTransform
        RandFlipd(keys=["image", lbl_key], prob=0.3,),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
    ]

    return Compose(monai_transforms) 

def inference_transforms(crop_size, lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], image_only=False),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            DivisiblePadd(keys=["image", lbl_key], k=2**5),   # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

def val_transforms(crop_size, lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], image_only=False),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])