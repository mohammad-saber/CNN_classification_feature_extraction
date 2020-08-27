
import torch.nn as nn
from torchvision import transforms
import os
import numpy as np
from classification import Predictor


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

model_name = 'resnet18'
classes_num = 2
input_img_size = (224, 224)   # [Height, Weight]
channel_num = 3   # 1 for grayscale and 3 for colored input image

result_folder = model_name + "_trial0" 

batch_size = 24

mean, std = np.array([0.1811, 0.1811, 0.1811]), np.array([0.1744, 0.1744, 0.1744])   # these values are for Mino Kogyo dataset. for other dataset, use mean and std of that dataset

# mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])   # for ImageNet dataset

# in case of 1-channel image
# mean, std = np.array([0.1811]), np.array([0.1744])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # example: '0, 1'   ; if you don't specify GPU, cuda will use all GPUs by default.


# Parameters for prediction
image_path = './dataset/1.jpg'
input_image_folder = './dataset/input_folder'   
dataset_path = './dataset/test'   


# -----------------------------------------------------------------------------
# Data Transform
# -----------------------------------------------------------------------------

# it's better to use transforms similar to validation dataset
test_transforms = transforms.Compose([
        transforms.Resize(input_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])  
                              

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
 
predictor = Predictor(model_name, classes_num, input_img_size, channel_num, result_folder,
                      batch_size, mean, std, test_transforms)


# -----------------------------------------------------------------------------
# Load pre-trained model
# -----------------------------------------------------------------------------

model = predictor.load_model()


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

output, predicted = predictor.predict_single_image(image_path, model)

output_all, predicted_all = predictor.predict_images(input_image_folder, model)

dataset_acc, dataset_loss, label_all, predicted_all, cm, cm_df = predictor.predict_dataset(dataset_path, model, criterion, plot_cm=True)
#print('Is model on Cuda? : ', next(model.parameters()).is_cuda, "\n")

