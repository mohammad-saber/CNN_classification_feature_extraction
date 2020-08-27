
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import os
import numpy as np
from classification import Trainer


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

model_name = 'resnet18'
ImageNet_pretrained = True
dataset_name = "dataset"
classes_num = 2
input_img_size = (224, 224)   # [Height, Weight]
channel_num = 3   # 1 for grayscale and 3 for colored input image

train_path = './dataset/train'
val_path = './dataset/val'   # validation (best model is saved based on the validation data)
result_folder = model_name + "_trial0" 

batch_size = 24
num_epochs = 20
learning_rate = 0.001
early_stopping_patience = 5   # parameter to contro early stopping
use_class_weight = True       # set True to use class weight (useful in case of imbalanced dataset)

mean, std = np.array([0.1811, 0.1811, 0.1811]), np.array([0.1744, 0.1744, 0.1744])   # these values are for Mino Kogyo dataset. for other dataset, use mean and std of that dataset

# mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])   # for ImageNet dataset

# in case of 1-channel image
# mean, std = np.array([0.1811]), np.array([0.1744])

best_model_selection = "acc"   # loss or acc ; Criterion to select the best model based on the validation dataset
save_model_all_epochs = False  # set True if you want to save models for all epochs 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # example: '0, 1'   ; if you don't specify GPU, cuda will use all GPUs by default.


# -----------------------------------------------------------------------------
# Data Transform
# -----------------------------------------------------------------------------

train_transforms = transforms.Compose([
        transforms.Resize(input_img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # after this transform, pixel values will be [0,1]
        transforms.Normalize(mean, std)
        ])
   
val_transforms = transforms.Compose([
        transforms.Resize(input_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])  

# transforms.Compose : Compose several transforms together.
# transforms.Resize : Resize the input PIL Image to the given size (It cannot be used with RandomResizedCrop).
# RandomResizedCrop : Crop the given PIL Image to random size (crop is not resize). This crop is finally resized to given size (It cannot be used with Resize). 
# RandomHorizontalFlip(p=0.5) : Horizontally flip the given PIL Image randomly with a given probability. Default value is 0.5.
# transforms.ToTensor() : Convert a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
# transforms.Normalize : Normalize a tensor image with mean and standard deviation (3-channel input image RGB, therefore 3 mean and std values)
# transforms.CenterCrop : Crops the given PIL Image at the center. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
# transforms.Grayscale(1) : Set the image channels for grayscale
# transforms.RandomRotation(180) : Rotate image randomly
# transforms.ColorJitter(brightness=0.2, contrast=0.2) : Change color attributes
# transforms.RandomAffine(10) : Apply affine transformation
                                   

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
 
trainer = Trainer(model_name, dataset_name, classes_num, input_img_size, channel_num,  
                  train_path, val_path, result_folder, batch_size, num_epochs, 
                  best_model_selection, save_model_all_epochs, mean, std, train_transforms, val_transforms)

class_weight = trainer.load_dataset()
trainer.logging_dataset_info()


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

model = trainer.build_model(ImageNet_pretrained)   # model in not on Cuda , check it with : next(model.parameters()).is_cuda

if use_class_weight:
    criterion = nn.CrossEntropyLoss(weight=class_weight)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)   # weight decay is for L2 penalty (regularization)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

model, train_acc_all, train_loss_all, val_acc_all, val_loss_all \
    = trainer.train(model, criterion, optimizer, scheduler, early_stopping_patience)
# model in not on Cuda

