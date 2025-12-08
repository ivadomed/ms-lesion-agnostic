import numpy as np
import monai.transforms as transforms


def multiply_by_negative_one(x):
    return np.min(x) + np.max(x) - x 


def aug_sqrt(img):
    # Compute original mean, std and min/max values
    img_min, img_max = img.min(), img.max()
    # Normalize
    img = (img - img.mean()) / img.std()
    img = np.interp(img, (img.min(), img.max()), (0, 1))
    # Transform
    img = np.sqrt(img)
    # Return to original range
    img = np.interp(img, (img.min(), img.max()), (img_min, img_max))
    return img


def aug_log(img):
    # Compute original mean, std and min/max values
    img_min, img_max = img.min(), img.max()
    # Normalize
    img = (img - img.mean()) / img.std()
    img = np.interp(img, (img.min(), img.max()), (0, 1))
    # Transform
    img = np.log(img + 1)
    # Return to original range
    img = np.interp(img, (img.min(), img.max()), (img_min, img_max))
    return img


def aug_exp(img):
    # Compute original mean, std and min/max values
    img_min, img_max = img.min(), img.max()
    # Normalize
    img = (img - img.mean()) / img.std()
    img = np.interp(img, (img.min(), img.max()), (0, 1))
    # Transform
    img = np.exp(img)
    # Return to original range
    img = np.interp(img, (img.min(), img.max()), (img_min, img_max))
    return img


def aug_sigmoid(img):
    # Compute original mean, std and min/max values
    img_min, img_max = img.min(), img.max()
    # Normalize
    img = (img - img.mean()) / img.std()
    img = np.interp(img, (img.min(), img.max()), (0, 1))
    # Transform
    img = 1 / (1 + np.exp(-img))
    # Return to original range
    img = np.interp(img, (img.min(), img.max()), (img_min, img_max))
    return img




def train_transforms(cfg):
    
    # define training transforms
    train_transforms = [
        
        # Preprocess
        transforms.LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RPI"),
        transforms.Spacingd(keys=["image", "label"], pixdim=cfg["pixdim"],mode=(2, 0)),
        # This crops the image around a foreground object of label with ratio pos/(pos+neg) (however, it cannot pad so keeping padding after)
        transforms.RandCropByPosNegLabeld(keys=["image", "label"],label_key="label",spatial_size=cfg["spatial_size"],
                                          pos=1,neg=0,num_samples=4,image_key="image",image_threshold=0,allow_smaller=True),
        # This resizes the image and the label to the spatial size defined in the config
        transforms.ResizeWithPadOrCropd(keys=["image", "label"],spatial_size=cfg["spatial_size"]),
        
        # Data augmentation
        # Random affine transform of the image
        transforms.RandAffined(keys=["image", "label"], mode=(2, 1), prob=0.9,
                    rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians
                    scale_range=(-0.2, 0.2),
                    translate_range=(-0.1, 0.1)),
        # Random elastic deformation
        transforms.Rand3DElasticd(keys=["image", "label"],sigma_range=(3.5, 5.5),magnitude_range=(25., 35.),
                                  prob=0.5,mode=['bilinear', 'nearest']),
        # Random simulation of low resolution 
        transforms.RandSimulateLowResolutiond(keys=["image"],zoom_range=(0.8, 1.5),prob=0.25),
        transforms.RandAdjustContrastd(keys=["image"],prob=0.5,gamma=(0.5, 3.)),
        transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
        transforms.RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),  # this is nnUNet's BrightnessMultiplicativeTransform
        transforms.RandGaussianNoised(keys=["image"],mean=0.0, std=0.1, prob=0.2),
        transforms.RandBiasFieldd(keys=["image"],coeff_range=(0.0, 0.5),degree=3, prob=0.3),
        transforms.RandShiftIntensityd(keys=["image"],offsets=0.1,prob=0.2,),
        
        # Applying functions
        # we add the multiplication of the image by -1
        transforms.RandLambdad(keys='image',func=multiply_by_negative_one,prob=0.2),
        transforms.RandLambdad(keys='image',func=aug_sqrt,prob=0.05),
        transforms.RandLambdad(keys='image',func=aug_log,prob=0.05),
        transforms.RandLambdad(keys='image',func=aug_exp,prob=0.05),
        transforms.RandLambdad(keys='image',func=aug_sigmoid,prob=0.05),

        # Transform image into its laplacian
        # transforms.LabelToContourd(keys=["image"], kernel_type='Laplace'),        
        
        # Normalize the intensity of the image
        transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
    ]

    return transforms.Compose(train_transforms)

def inference_transforms(crop_size, lbl_key="label"):
    return transforms.Compose([
            transforms.LoadImaged(keys=["image", lbl_key], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            transforms.Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            transforms.Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            transforms.ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            transforms.DivisiblePadd(keys=["image", lbl_key], k=2**5),   # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

def val_transforms(crop_size, lbl_key="label", pad_mode="zero"):
    return transforms.Compose([
            transforms.LoadImaged(keys=["image", lbl_key], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            transforms.Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            transforms.Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            transforms.ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,
                                            mode="constant" if pad_mode == "zero" else pad_mode),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])