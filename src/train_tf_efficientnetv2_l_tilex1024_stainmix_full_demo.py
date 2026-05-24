import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
ROOT_DIR = os.path.join(DATASET_DIR, 'processed', 'ubc-mini-tiles-2048-scale-0-5')  # 训练 tile 数据目录
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, 'train_images')
STAIN_DIR = os.path.join(DATASET_DIR, 'ubc-tilesx1024-stain3-data')

# https://huggingface.co/timm/tf_efficientnetv2_l.in21k_ft_in1k/resolve/main/pytorch_model.bin
checkpoint_path = os.path.join(DATASET_DIR, 'hub', 'tf_efficientnetv2_l.in21k_ft_in1k.bin')

SAVE = os.path.join(BASE_DIR, 'save', 'full_demo')  # 保存全量 demo 模型路径
os.makedirs(SAVE, exist_ok=True)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt

# For data manipulation
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision
import torchmetrics
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
scaler = torch.cuda.amp.GradScaler()

all_wsi_images = sorted(glob.glob(os.path.join(TRAIN_IMAGES_DIR, '*', '*.png')))
#all_wsi_images.remove('/root/data/UBC/ubc-tiles-with-masks-2048px-scale-0-25/train_images/12442/00293_40-10.png')
#all_wsi_images = sorted(glob.glob('/root/data/UBC/ubc-tiles-stain-data2/stain/*/*.png'))
train = pd.read_csv(f"{ROOT_DIR}/train.csv")
train.loc[train['image_id']==15583, 'label']='MC'
df = pd.DataFrame(all_wsi_images,columns=['file_path'])
print(df.shape)
df['image_id'] = df['file_path'].map(lambda x:x.split('/')[-2]).map(int)
df['label'] = df['image_id'].map(train.set_index(['image_id'])['label'])
label2index = {'CC':0, 'EC':1, 'HGSC':2, 'LGSC':3, 'MC':4}
index2label = {0:'CC', 1:'EC', 2:'HGSC', 3:'LGSC', 4:'MC'}
df['label'] = df['label'].map(label2index)
df
#http://36.137.227.27:11112/edit/data/UBC/hub/tf_efficientnet_b6.ns_jft_in1k.bin

CONFIG = {
    "seed": 42,
    "epochs": 15,
    "img_size": 1024,
    "model_name": "tf_efficientnetv2_l.in21k_ft_in1k",
    "checkpoint_path" : checkpoint_path,
    "pretrained" : None,
    "num_classes": 5,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "learning_rate": 5e-5,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-7,
    "T_max": 500,
    "weight_decay": 1e-6,
    "fold" : "full",
    "n_fold": 1,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

CLASS_WEIGHTS = Counter(df["label"].values)
CLASS_WEIGHTS = [ df.shape[0] / CLASS_WEIGHTS[i] for i in sorted(df["label"].unique()) ]
CLASS_WEIGHTS = [ val / sum(CLASS_WEIGHTS) for val in CLASS_WEIGHTS ]
CLASS_WEIGHTS

CONFIG['T_max'] = math.ceil(df.shape[0] * CONFIG['epochs'] / CONFIG['train_batch_size'])
CONFIG['T_max']

print("Full demo training with all tiles:", df.shape[0])
print(df.head())
class UBCDataset(Dataset):
    def __init__(self, df, transforms=None, train=False):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        self.train = train
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        p = np.random.random()
        if self.train and p > 0.3 and os.path.isdir(STAIN_DIR):
            stain_path = img_path.replace(TRAIN_IMAGES_DIR, STAIN_DIR)
            if os.path.exists(stain_path):
                img_path = stain_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        } 
transforms_train = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.75),
    A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.75),

    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.GaussianBlur(blur_limit=3),
        A.GaussNoise(std_range=(math.sqrt(3.0) / 255.0, math.sqrt(9.0) / 255.0)),
    ], p=0.5),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.),
        A.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.5),

    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(int(CONFIG['img_size'] * 0.4), int(CONFIG['img_size'] * 0.4)),
        hole_width_range=(int(CONFIG['img_size'] * 0.4), int(CONFIG['img_size'] * 0.4)),
        p=0.5,
    ),
    A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
    ToTensorV2(),
])

transforms_valid = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
    ToTensorV2(),
])

data_transforms = {
    "train": transforms_train,

    "valid": transforms_valid,
}
model = timm.create_model(CONFIG['model_name'], pretrained=0, checkpoint_path=None, num_classes=5)#UBCModel(CONFIG['model_name'], CONFIG['num_classes'], checkpoint_path=CONFIG['checkpoint_path'])
state_dict = torch.load(CONFIG['checkpoint_path'])
#del state_dict['fc.weight']
#del state_dict['fc.bias']
del state_dict['classifier.weight']
del state_dict['classifier.bias']
model.load_state_dict(state_dict, strict=False)
model.to(CONFIG['device']);
if CONFIG["pretrained"] is not None:
    model.load_state_dict(torch.load(CONFIG["pretrained"]))
    print('load pretrain:', CONFIG["pretrained"])
def criterion(outputs, labels):
    return nn.CrossEntropyLoss( weight = torch.tensor(CLASS_WEIGHTS).cuda() )(outputs, labels)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_acc  = 0.0
    running_recall = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CONFIG['n_accumulate']
            
        #loss.backward()
        scaler.scale(loss).backward()
        
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        _, predicted = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
        acc = torch.sum( predicted == labels )
        recall_nn = torchmetrics.Recall(task="multiclass", average='macro', num_classes=CONFIG["num_classes"]).cuda()
        recall = recall_nn(predicted, labels)

        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        running_recall += (recall.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        epoch_recall = running_recall / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc, Train_Recall=epoch_recall,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_acc, epoch_recall

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
        acc = torch.sum( predicted == labels )

        recall_nn = torchmetrics.Recall(task="multiclass", average='macro', num_classes=CONFIG["num_classes"]).cuda()
        recall = recall_nn(predicted, labels)

        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        running_recall += (recall.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        epoch_recall = running_recall / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc, Valid_Recall=epoch_recall,
                        LR=optimizer.param_groups[0]['lr'])

    gc.collect()

    return epoch_loss, epoch_acc, epoch_recall

def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    
    start = time.time()
    best_train_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_acc, train_epoch_recall = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader,
                                           device=CONFIG['device'], epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Train Accuracy'].append(train_epoch_acc)
        history['Train Recall'].append(train_epoch_recall)
        history['lr'].append( scheduler.get_lr()[0] )

        if train_epoch_loss <= best_train_loss:
            print(f"Train Loss Improved ({best_train_loss} ---> {train_epoch_loss})")
            best_train_loss = train_epoch_loss
            PATH = f"{SAVE}/{CONFIG['model_name']}_bestTrainLoss_imgsize_{CONFIG['img_size']}_{CONFIG['fold']}.pt"
            torch.save(model.state_dict(), PATH)
            print(f"Best train-loss model saved: {PATH}")

        if epoch == num_epochs:
            PATH = f"{SAVE}/{CONFIG['model_name']}_final_ep{epoch}_imgsize_{CONFIG['img_size']}_{CONFIG['fold']}.pt"
            torch.save(model.state_dict(), PATH)
            print(f"Final model saved: {PATH}")
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Train Loss: {:.4f}".format(best_train_loss))

    return model, history

def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3, 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def prepare_loader(df):
    df_train = df.reset_index(drop=True)

    train_dataset = UBCDataset(df_train, transforms=data_transforms["train"], train=True)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=8, shuffle=True, pin_memory=False, drop_last=False)

    return train_loader

train_loader = prepare_loader(df)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)

model, history = run_training(model, optimizer, scheduler,
                              device=CONFIG['device'],
                              num_epochs=CONFIG['epochs'])
