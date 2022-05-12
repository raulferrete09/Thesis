from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandZoomd,
    RandGridDistortiond,
)
from monai.handlers.utils import from_engine
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import time
import glob
import tqdm
import numpy as np
from IPython.core.debugger import set_trace
from torch.utils.tensorboard import SummaryWriter
import random
from monai.data.utils import pad_list_data_collate
import pdb
import torch.nn as nn
import nibabel as nib
import time
from MDANet import MDA_Net

data_dir = "/home/users/jvilaca/raul/3_2_CBIS_Croped_padding"
print(data_dir)

#Train
train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]


#Val
val_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesVl", "*.nii.gz")))
val_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsVl", "*.nii.gz")))
val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(val_images, val_labels)
]

#Test
test_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
test_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTs", "*.nii.gz")))
test_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(test_images, test_labels)
]

set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandScaleIntensityd(keys="image", factors=0.5, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.5, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys="image", prob=0.25),
        RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.5,2.5)),
        RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=1, max_zoom=1.3),
        RandGridDistortiond(keys=["image", "label"], prob=0.5, distort_limit=(-0.2,0.2)),

        EnsureTyped(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=1.0, num_workers=4)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=0)


val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
# val_ds = Dataset(data=val_files, transform=val_transforms)

val_loader = DataLoader(val_ds, batch_size=5, num_workers=4)


test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=0)
# val_ds = Dataset(data=val_files, transform=val_transforms)

test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

DEVICE = torch.device("cuda:0")
best_metric = -1
best_metric_epoch = -1

def check_accuracy(loader, model, device, loss_fn):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch_data in loader:
            inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device).unsqueeze(1),
            )
            dims = np.shape(inputs)        
            inputs = torch.reshape(inputs,[dims[0],1,dims[2],dims[3]])
            # print(np.shape(inputs))
            labels = torch.reshape(labels,[dims[0],1,dims[2],dims[3]])
            # print(np.shape(labels))
            #x = x.to(device)
            #y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(inputs))
            preds = (preds > 0.5).float()
            num_correct += (preds == labels).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * labels).sum()) / (
                (preds + labels).sum() + 1e-8
            )
            
             # forward
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                loss = loss_fn(predictions, labels)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    dice = dice_score/len(loader)

    model.train()

    return dice, loss

model = MDA_Net(img_ch=1, output_ch=1).to(DEVICE)
#loss_fn = nn.BCEWithLogitsLoss()
loss_fn = DiceCELoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# if LOAD_MODEL:
#     checkpoint = torch.load("my_checkpoint.pth.tar")
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])

# pdb.set_trace()
check_accuracy(val_loader, model, DEVICE, loss_fn)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter()

model_path = "/home/users/jvilaca/raul/Methods/MDA-UNet/Jan04_18-32-41"
model_name = "best_metric_model.pth"

checkpoint = torch.load(os.path.join(model_path , model_name), map_location='cuda')
model.load_state_dict(checkpoint['model'])

model.eval()
i = 0
sig=nn.Sigmoid()
with torch.no_grad():
    for test_data in test_loader:
        test_inputs, test_labels = (
            test_data["image"].to(device=DEVICE),
            test_data["label"].to(device=DEVICE),
        )
        dimsTest = np.shape(test_inputs)
        test_inputs = torch.reshape(test_inputs,[dimsTest[0],1,dimsTest[2],dimsTest[3]])
        test_labels = torch.reshape(test_labels,[dimsTest[0],1,dimsTest[2],dimsTest[3]])
        tic=time.process_time()
        kk = model(test_inputs)
        toc=time.process_time()
        print(toc-tic)
        kk = sig(kk)
#         kk=kk[0]
        kk = torch.where(kk>0.90, 1, 0)
        kk = np.expand_dims(kk[0,0,:,:].cpu().detach().numpy(), axis = 2) 
        name = test_files[i]['image']
        masknii = nib.load(name)  
        m = masknii.affine
        header = masknii.header
        predicted = nib.Nifti1Image(kk, m, header)
        nib.save(predicted, pathSave + name.split('/')[-1])
        i = i + 1

