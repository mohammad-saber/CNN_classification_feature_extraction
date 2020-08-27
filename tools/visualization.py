
# Visualization tools

import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
from torchvision import datasets
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd


def show_image_batch(data_loader, path, mean, std, class_names):
    """
    Visualize training images, one batch. This function is called before training. 
    data_loader: usually train_loader is passed. Because transforms are all applied on train dataset.
    """
    
    # Get a batch of training data , iteration through train dataset and get the next value
    images, classes = next(iter(data_loader))

    title=[class_names[x] for x in classes]

    # Make a grid from batch. It concatenates all images into one Tensor. 
    # nrow : Number of images displayed in each row of the grid. 
    image = torchvision.utils.make_grid(images, nrow=8)
     
    # Input is Image but it was converted into PyTorch tensor by data_transforms. We need to re-convert tensor into numpy array before image show. 
    # Convert PyTorch tensor into numpy array , put the 1st 2nd dimensions in the 0 and 1st dimentions. 
    # and put the 0 dimension into 3rd dimention. Because in PyTorch, Image Tensor is (C x H x W).
    image = image.numpy().transpose((1, 2, 0))  
    
    image = std * image + mean  # In data_transforms, we used transforms.Normalize. This code is de-normalization before image show.
    image = np.clip(image, 0, 1)  # Clip (limit) the values in an array. Otherwise, there might be warning. 
    
    fig = plt.figure()
    plt.imshow(image)
    plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated
    fig.savefig(path)
    plt.close()


def show_image(data_folder, path, mean, std, image_num=6, row = 1, title = "Data Augmentation Result"):
    """
    Visualize training images. This function is called before training. 
    data_folder: usually train_folders is passed. Because transforms are all applied on train dataset.
    images_num: number of images you would like to show. 
    path: path to save figure
    row: number of rowa in image grid
    """
    class_names = data_folder.classes   # name of classes

    total_img = []   # List that keeps all images, every image is a Numpy array.
    labels = []
    for num in range(image_num):
        image = np.array(data_folder.__getitem__(num)[0]).transpose((1, 2, 0))   # data_folder.__getitem__(num)[0] is the image
        image = std * image + mean  # In data_transforms, we used transforms.Normalize. This code is de-normalization before image show.
        image = np.clip(image, 0, 1)  # Clip (limit) the values in an array. Otherwise, there might be warning.
        total_img.append(image) 
        labels.append(class_names[data_folder.__getitem__(num)[1]])   # data_folder.__getitem__(num)[1] is the index of class
        
    fig = plt.figure()
    plt.axis("off")   # hide axis
    plt.title(title, fontweight='bold')
    for n, (image, label) in enumerate(zip(total_img, labels)):
        ax = fig.add_subplot(row, np.ceil(image_num / float(row)), n + 1)
        if image.ndim == 2:
            plt.gray()   # Set the colormap to "gray"
        plt.axis("off")
        plt.imshow(image)
        ax.set_title(label)
    fig.set_size_inches(fig.get_size_inches() * int(image_num/6))
    plt.savefig(path)
    plt.close()
    

def plot_graph(data_dict, title, x_label, y_label, path, y_scale_log=False):
    '''
    Plots the accuracy and loss vs epoch for train and test data (after training).
    This function can plot accuracy or loss for several models. 

    data_dict: a dictionary containing accuracy or loss of the trained models. 
    Every model has a key in this dictionary. In fact, key is the model name. 
    
    path: path to save graph as an image file
    '''
    
    fig = plt.figure()
    for model in data_dict.keys():   # loop over Keys. Key is the model name. 
        plt.plot(list(range(1, len(data_dict[model])+1)), data_dict[model], label=model)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_scale_log: plt.yscale("log")
    plt.title(title)
    plt.legend()
    fig.savefig(path, dpi=200)
    plt.close()
    

def plot_confusion_matrix(cm, class_names, save_path, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
       
    if normalize:
        # axis = 1: rows ; [:, np.newaxis] adds one more dimension to the 1D array
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    ax.xaxis.tick_top()   # send x-labels to the top of the plot

    fmt = '.2f' if normalize else 'd'   # format to display numbers
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] >= thresh else "black")   # j is x direction, i is y direction

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_prediction(dataset_path, save_path, model, model_name, test_transforms, 
                         mean, std, cuda, images_num=12, row=2, title = 'Model Prediction'):
    '''
    Display predictions for a few images, this function is called after training
    dataset_path: usually test dataset path is passed. 
    images_num: number of images you would like to show. 
    save_path: path to save figure
    row: number of rows in image grid
    '''
       
    dataset_folders = datasets.ImageFolder(dataset_path, test_transforms) 
    dataset_loader = torch.utils.data.DataLoader(dataset_folders, batch_size=images_num, shuffle=True)
    class_names = dataset_folders.classes  # name of classes

    model.eval()

    images, labels = next(iter(dataset_loader))
    
    fig = plt.figure()
    plt.axis("off")   # hide axis
    plt.title(title, fontweight='bold')

    with torch.no_grad():
        for i in range(images.shape[0]):   # images.size()[0] is Batch Size
            
            image_tensor = images[i,:,:,:]              # shape : [channel_num, H, W]
            image_tensor = image_tensor[None, :, :, :]  # shape : [1, channel_num, H, W]
            
            if cuda:
                model = model.cuda()
                image_tensor = image_tensor.cuda()
            
            if model_name == 'vgg16':
                output = model(image_tensor)[-1]   
            else:
                output = model(image_tensor)
            
            _, preds = torch.max(output, 1)

            image = np.array(images[i,:,:,:]).transpose((1, 2, 0))   # [H, W, channel_num] 
            image = std * image + mean  # In data_transforms, we used transforms.Normalize. This code is de-normalization before image show.
            image = np.clip(image, 0, 1)  # Clip (limit) the values in an array. Otherwise, there might be warning.

            ax = fig.add_subplot(row, np.ceil(images_num / float(row)), i + 1)
            if image.ndim == 2:
                plt.gray()   # Set the colormap to "gray"
            plt.axis("off")
            plt.imshow(image)
            ax.set_title(class_names[preds])
    
        plt.savefig(save_path)
        plt.close()
    
    model = model.cpu()   # some functions require model on CPU. 


def pretty_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted '+ l for l in class_names]
    df = pd.DataFrame(cm, index=class_names, columns=pred_labels)
    return df

