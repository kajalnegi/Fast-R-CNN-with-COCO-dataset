#!/usr/bin/env python
# coding: utf-8

from pycocotools.coco import COCO
from engine import train_one_epoch, evaluate

import imgaug as ia
import imgaug.augmenters as iaa
import math

from skimage import io
import torch , json
import numpy as np
import random , os
from copy import deepcopy

# # load data and annotations
import pandas as pd
from PIL import Image , ImageDraw
from collections import Counter

from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
from torch.optim import lr_scheduler

import torch
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from collections import defaultdict
import json
from pathlib import Path
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data import Dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

# Define your transformation for training and validation (modify as per your requirements)
def get_transform():
    custom_transforms = []
    #custom_transforms.append(torchvision.transforms.Resize((256, 256)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)



def collate_fn(batch):
    batch = [sample for sample in batch if sample[0].shape == batch[0][0].shape]
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



class CocoDataset(Dataset):
    """PyTorch dataset for COCO annotations."""

    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        width, height = img.size
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        num_obj = 0
        #targets = []
  
        for i in range(num_objs):
            
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = min(xmin + coco_annotation[i]['bbox'][2], width)
            ymax = min(ymin + coco_annotation[i]['bbox'][3], height)
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            #targets.append(coco_annotation[i]['category_id']-1)
            num_obj = num_obj+1
            #targets.append(trg)
        if  not boxes:
            boxes = torch.zeros(0, 4, dtype=torch.float32)#torch.zeros(0,4)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #print(len(boxes)==len(targets))
        # Labels 
        #ground_truth_labels = [ann['category_id'] for ann in anns]
        labels = torch.ones((num_obj,), dtype=torch.int64)#torch.as_tensor(targets, dtype=torch.int64)
        # Tensorise img_id
        img_id = img_id#torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_obj):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_obj,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        
        my_annotation["boxes"] = boxes 
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        #print('my_annotation["boxes"] ', my_annotation["boxes"])
        if self.transforms is not None:
            img = self.transforms(img)
            #my_annotation
        return img, my_annotation

    def __len__(self):
        return len(self.ids)


batch_size = 8
# Create DataLoaders for training and validation

train_dataset = CocoDataset(root='/home/kajalnegi/raid/NNDL/cocodataset/train2017/',
                          annotation='/home/kajalnegi/raid/NNDL/cocodataset/train_annotations/annotations/instances_train2017.json',
                          transforms=get_transform()
                          )

#background_only = BackGroundOnly(num_images=100000, transforms=get_transform()
                          #)

val_dataset = CocoDataset(root='/home/kajalnegi/raid/NNDL/cocodataset/val2017/',
                          annotation='/home/kajalnegi/raid/NNDL/cocodataset/train_annotations/annotations/instances_val2017.json',
                          transforms=get_transform()
                          )

#train_dataset = ConcatDataset([train_dataset, background_only])
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn,
                                          drop_last=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn,
                                          drop_last=True)



# Load the pre-trained COCO model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device, ", device)

num_classes = 2  # 1 class (person) + background

num_epochs = 268 # Choose the number of epochs you want to train for
lrf = 0.005
# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
params = [p for p in model.parameters() if p.requires_grad]

# Set up optimizer
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
model.to(device)

best_loss = 100
df = pd.DataFrame(columns=['epoch', 'val_loss_classifier', 'val_loss_box_reg', 'val_loss_objectness',
'train_loss_classifier', 'train_loss_box_reg','train_loss_objectness', ])



for epoch in range(num_epochs):
    print("epoch ", epoch)
    
    # train for one epoch, printing every 10 iterations
    loss_classifier, loss_box_reg, loss_objectness,loss_rpn_box_reg, avg_total_loss = train_one_epoch(
        model, optimizer, train_dataloader, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()

    
    # evaluate on the test dataset
   
    coco_evaluator, val_loss_classifier, val_loss_box_reg, val_loss_objectness,val_loss_rpn_box_reg, avg_val_loss, pr_metr = evaluate(
        model, val_loader, device=device)
    
    row_dict = dict()
    row_dict['epoch'] = epoch
    row_dict['val_loss_classifier']  = val_loss_classifier
    row_dict['val_loss_box_reg']  = val_loss_box_reg
    row_dict['val_loss_objectness']  = val_loss_objectness
    row_dict['val_loss_rpn_box_reg']  = val_loss_rpn_box_reg
    row_dict['avg_total_val_loss']  = avg_val_loss
    row_dict['train_loss_classifier']  = loss_classifier
    row_dict['train_loss_box_reg']  = loss_box_reg
    row_dict['train_loss_objectness']  = loss_objectness
    row_dict['train_loss_rpn_box_reg']  = loss_rpn_box_reg
    row_dict['avg_total_train_loss']  = avg_total_loss
    
    row_dict.update(pr_metr)
    df = pd.concat([df,  pd.DataFrame.from_records([row_dict])], ignore_index=True)
    
    df.to_csv('fastrcnn_train_val_loss_1.csv', index=False)
    if best_loss >= loss_classifier+loss_box_reg+loss_objectness:
        torch.save(model, 'fastrcnn_best_1.pth')
        best_loss = loss_classifier+loss_box_reg+loss_objectness
    torch.save(model, 'fastrcnn_last_1.pth')

