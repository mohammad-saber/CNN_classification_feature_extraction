
import torch
import torch.nn as nn
from torchvision import datasets, models
import copy, os, time
import numpy as np
from collections import OrderedDict
from PIL import Image
import pandas as pd
from tools.logger import save_txt, save_csv, save_excel
from tools.visualization import show_image_batch, show_image, plot_graph, visualize_prediction, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from collections import Counter


class Trainer():
    
    def __init__(self, model_name, dataset_name, classes_num, input_img_size, channel_num, 
                 train_path, val_path, result_folder, batch_size, num_epochs, 
                 best_model_selection, save_model_all_epochs, mean, std, train_transforms, val_transforms):
        
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.classes_num = classes_num
        self.input_img_size = input_img_size
        self.channel_num = channel_num
        self.train_path = train_path
        self.val_path = val_path
        self.result_folder = result_folder
        self.batch_size = batch_size
        self.num_epochs = num_epochs        
        self.best_model_selection = best_model_selection        
        self.save_model_all_epochs = save_model_all_epochs
        self.mean = mean
        self.std = std
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        model_name_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'] + \
                  ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'] + \
                  ['alexnet'] + ['densenet121', 'densenet161', 'densenet169', 'densenet201'] + ['googlenet'] + ['mobilenet_v2'] + ['mnasnet0_5', 'mnasnet1_0']

        if self.model_name not in model_name_list:
            raise Exception('model_name is not valid')
        
        self.GPU_num = torch.cuda.device_count()

        self.cuda = torch.cuda.is_available()

        self.result_main_path = "./trained_models/{}".format(self.result_folder)
        if not os.path.exists(self.result_main_path):
            os.mkdir(self.result_main_path)  
        
        self.checkpoint_dir = os.path.join(self.result_main_path, "checkpoint").replace('\\', '/')
        if self.save_model_all_epochs:
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)

        self.txt_path = os.path.join(self.result_main_path, "summary.txt").replace('\\', '/')
        
    
    def load_dataset(self):
        
        # "datasets.ImageFolder" is data loader for image. Transforms are applied here.
        # Images are loaded as PIL images. Then, transforms are applied on PIL images.     
        self.train_folders = datasets.ImageFolder(self.train_path, self.train_transforms)
        self.val_folders = datasets.ImageFolder(self.val_path, self.val_transforms)
        print('\nShape of image after loading: ', self.train_folders[0][0].shape, '\n')

        # provide an iterable over the given dataset. Every iteration gives a batch of dataset. 
        # "num_workers" parameter is not used to avoid "broken pipe" error in Windows
        self.train_loader = torch.utils.data.DataLoader(self.train_folders, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_folders, batch_size=self.batch_size, shuffle=False)
        
        # check data size
        trainiter = iter(self.train_loader)
        inputs, labels = next(trainiter)
        #print('\ninput tensor size : ', inputs.shape, '\nlabel tensor size : ', labels.shape, '\n')
        
        self.train_img_num = len(self.train_folders)   # No. of training images
        self.val_img_num = len(self.val_folders)     # No. of val images
        #self.classes_num = len(self.train_folders.classes)
        self.class_names = self.train_folders.classes  # name of classes
        #print('No. of images in train and val dataset : ', self.train_img_num, ' , ', self.val_img_num)

        # No. of images in every class
        image_class = Counter(self.train_folders.targets)
        
        class_weight = []   # class weight used for imbalanced dataset       
        for key in image_class:
            class_weight.append(1/image_class[key])
        
        if self.cuda:
            class_weight = torch.FloatTensor(class_weight).cuda()
        else:
            class_weight = torch.FloatTensor(class_weight) 
        
        return class_weight    
        
    
    def logging_dataset_info(self):
        '''
        Logging information, Visualization
        '''        
        save_txt(self.txt_path, "", "Dataset Name : {} , number of classes : {} , input image size : {} , channel number : {}"
                        .format(self.dataset_name, self.classes_num, self.input_img_size, self.channel_num), "")
        
        save_txt(self.txt_path, "No. of images in train and val dataset : {} , {}".format(self.train_img_num, self.val_img_num), "")
        
        save_txt(self.txt_path, "Model used for training : {}".format(self.model_name), "")
        
        save_txt(self.txt_path, "No. of GPU used for training : {}".format(self.GPU_num), "")
        
        save_txt(self.txt_path, "="*50, "")
        
        
        # Visualize a few training images after transform, so as to understand the data augmentations.
        show_image_batch(self.train_loader, os.path.join(self.result_main_path, "Augmentation_Result_Batch.png").replace('\\', '/'), self.mean, self.std, self.class_names)   # show images for one batch after transform
        
        show_image(self.train_folders, os.path.join(self.result_main_path, "Augmentation_Result.png").replace('\\', '/'),
                   self.mean, self.std, image_num=6, row = 5, title = "Data Augmentation Result")
    
    
    def train(self, model, criterion, optimizer, scheduler, early_stopping_patience):
        
        if self.cuda:
            if self.GPU_num > 1:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model.cuda()
        #print('Is model on Cuda? : ', next(model.parameters()).is_cuda, "\n")

        save_txt(self.txt_path, "No. of model parameters [trainable parameters, total] : " + str(self.count_parameters(model)), "")
        save_txt(self.txt_path, "="*50, "")
        save_txt(self.txt_path, "EPOCH, train_acc, train_loss, val_acc, val_loss")

        # dictionary to keep accuracy and loss of train and val for all models, every model has a key. 
        train_acc_dict, train_loss_dict = {}, {}
        val_acc_dict, val_loss_dict = {}, {}
        

        train_acc_all = []
        train_loss_all = []
        val_acc_all = []
        val_loss_all = []
       
        # model.state_dict() : Returns a dictionary containing a whole state of the module (w , bias).
        best_model_wts = copy.deepcopy(model.state_dict())   # keep the best values of weights
        best_acc = 0.0        # initial value
        best_loss = 10000.0   # initial value
        epochs_no_improve = 0 # this parameter is used for early stopping. No. of epochs without improvement in validation loss.

        training_time = time.time()

        for epoch in range(1, self.num_epochs+1):
    
            print('\n', 'Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 15)
            
            ########## Training section ##########
            model.train()   # Set model to training mode
            total_loss = 0.0
            total_correct = 0.0   # No. of correct predictions
            total_img = 0.0
    
            for batch_index, (img, label) in enumerate(self.train_loader):
    
                if self.cuda:
                    img = img.cuda()
                    label = label.cuda()
                img.requires_grad = True   # by default, requires_grad is False. But, without this line, it still works
    
                optimizer.zero_grad()   # zero the parameter gradients
                
                output = model(img)
    
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
    
                _, predicted = torch.max(output.data, 1)   # output.data is similar to output but "requires_grad" = False. In other words, only the value or data of Tensor. Data-type of "output.data" is still Tensor. 
    
                total_loss += loss.item()   # .item() converts Tensor to python float
                total_correct += (predicted == label.data).sum().item()
                total_img += label.size(0)   # No. of images in the epoch or batch_size
    
                print(" Train : EPOCH [{}] BATCH [{}] ACC [{:.4}] loss [{:.4}]".format(epoch, batch_index, 
                      (predicted == label.data).sum().item() / label.size(0), loss.item()))
    
            train_epoch_acc = total_correct / total_img
            train_epoch_loss = total_loss / total_img
    
            if self.save_model_all_epochs:
                torch.save(model.state_dict(),
                           os.path.join(self.checkpoint_dir, '{}-model-epoch{}.pth'.format(self.model_name, epoch)))
    
            train_acc_all.append(train_epoch_acc)
            train_loss_all.append(train_epoch_loss)
    
    
            ########## Validation section ##########
            model.eval()
            total_loss = 0.0
            total_correct = 0.0
            total_img = 0.0
    
            with torch.no_grad():   # temporarily set all the requires_grad flags to false (no gradient calculations).
                
                for batch_index, (img, label) in enumerate(self.val_loader):
                    
                    if self.cuda:
                        img, label = img.cuda(), label.cuda()
                    img.requires_grad = False
                    label.requires_grad = False
                    
                    output = model(img)
                  
                    loss = criterion(output, label)
                    _, predicted = torch.max(output.data, 1)
                    total_loss += loss.item()
                    total_correct += (predicted == label.data).sum().item()
                    total_img += label.size(0)   # No. of images in the epoch or batch_size
    
            val_epoch_acc = total_correct / total_img
            val_epoch_loss = total_loss / total_img
    
            print(" Val : EPOCH [{}] ACC [{:.4}] loss [{:.4}]".format(epoch, val_epoch_acc, val_epoch_loss))
    
            text_log = [str(x) for x in [epoch, train_epoch_acc, train_epoch_loss, val_epoch_acc, val_epoch_loss]]
            text_log = "\t".join(text_log)
            save_txt(self.txt_path, text_log)
    
            val_acc_all.append(val_epoch_acc)
            val_loss_all.append(val_epoch_loss)
            
            scheduler.step()   # Update the value of learning rate
    
    
            # save the best model
            if self.best_model_selection == "acc" and val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
            if self.best_model_selection == "loss" and val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # check early stopping
            if val_epoch_loss < best_loss:
                epochs_no_improve = 0
            else:
                epochs_no_improve +=1
                if epochs_no_improve == early_stopping_patience:
                    print('early stopping in epoch {}'.format(epoch))
                    break
            
            
        model.load_state_dict(best_model_wts)   # load the best weights after all epochs into model, return model
        
        
        training_time = time.time() - training_time

        # Save the best model into the result folder.
        model = model.cpu()   # it is recommended to save on CPU to avoid unpredictable error
        #print('Is model on Cuda? : ', next(model.parameters()).is_cuda)
        torch.save(model.state_dict(), os.path.join(self.result_main_path, '{}-bestmodel.pth'.format(self.model_name)).replace('\\', '/'))
        
        
        # Add key-value to dictionary. Key is the model name. Value is accuracy or loss for all epochs. 
        # Dictionary can hold all models' results. Then, it is used for plotting the results. 
        train_acc_dict.setdefault(self.model_name, train_acc_all)
        train_loss_dict.setdefault(self.model_name, train_loss_all)
        val_acc_dict.setdefault(self.model_name, val_acc_all)
        val_loss_dict.setdefault(self.model_name, val_loss_all)
        
        
        save_txt(self.txt_path, "", "="*50, "") 
        save_txt(self.txt_path, "Total time (training and val): {} h : {} m : {:05.2f} s".format(
                        training_time // 3600, (training_time - training_time // 3600) // 60, training_time % 60), "")
        save_txt(self.txt_path, "="*50, "") 
        
        print("\n", '-' * 15)
        print("\n", "Training and val is over. total time: {} h : {} m : {:05.2f} s".format(
                        training_time // 3600, (training_time - training_time // 3600) // 60, training_time % 60), "\n")
        
        
        plot_graph(train_acc_dict, "train accuracy", "epoch", "acc", os.path.join(self.result_main_path, "train_acc.png"))
        plot_graph(train_loss_dict, "train loss", "epoch", "loss", os.path.join(self.result_main_path, "train_loss.png"))
        plot_graph(val_acc_dict, "val accuracy", "epoch", "acc", os.path.join(self.result_main_path, "val_acc.png"))
        plot_graph(val_loss_dict, "val loss", "epoch", "loss", os.path.join(self.result_main_path, "val_loss.png"))
        
        result_excel_log = {"train_acc":train_acc_all, "train_loss":train_loss_all,
                            "val_acc":val_acc_all, "val_loss": val_loss_all}
        save_excel(os.path.join(self.result_main_path, "train_summary.xlsx").replace('\\', '/'),
                   [("acc-loss", result_excel_log)])
                
        
        return model, train_acc_all, train_loss_all, val_acc_all, val_loss_all


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad), \
               sum(p.numel() for p in model.parameters())
               # numel(): calculates number of elements in PyTorch tensor
       
        
    def build_model(self, ImageNet_pretrained):
        '''
        By default, model is built on training status (model.training = True)
        '''
        
        # old method, this methos only works for one model. 
    #    if self.model_name == 'resnet18':
    #        if not ImageNet_pretrained:
    #            return models.resnet18(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
    #        else:
    #            model = models.resnet18(pretrained=ImageNet_pretrained, num_classes=1000)
    #            num_ftrs = model.fc.in_features   # No. of input features to fully connected layer
    #            model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
    #            return model
        
        if self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.fc.in_features   # No. of input features to fully connected layer
                model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[6].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[6] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['alexnet']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[6].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[6] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier.in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        '''
        if self.model_name in ['inception_v3']:
            # This network is unique because it has two output layers when training. The second output is an auxiliary output. When testing, we only consider the primary output.
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.fc.in_features   # No. of input features to fully connected layer, last linear layer
                num_ftrs_aux = model.AuxLogits.fc.in_features   # auxiliary output
                model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                model.AuxLogits.fc = nn.Linear(num_ftrs_aux, self.classes_num)   # auxiliary output
                return model
        '''
        
        if self.model_name in ['googlenet']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.fc.in_features   # No. of input features to fully connected layer, last linear layer
                model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['mobilenet_v2']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[1].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[1] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['mnasnet0_5', 'mnasnet1_0']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[1].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[1] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    

    def build_model_feature(self, model):
        '''
        Build a model for feature extraction based on the structure of pre-trained model
        The last linear layer will be removed.
        '''
        if self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
            last_layer_removed = list(model.children())[:-1]   # all layers except last layer
            return torch.nn.Sequential(*last_layer_removed)    # make a new model based on last_layer_removed
    
        if self.model_name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            model_feature = copy.deepcopy(model)        
            last_layer_removed = list(model_feature.classifier.children())[:-1]   # all layers except last layer
            new_classifier = nn.Sequential(*last_layer_removed)
            model_feature.classifier = new_classifier   # make a new model based on new classifier
            return model_feature  
    
        if self.model_name in ['alexnet']:
            model_feature = copy.deepcopy(model)        
            last_layer_removed = list(model_feature.classifier.children())[:-1]   # all layers except last layer
            new_classifier = nn.Sequential(*last_layer_removed)
            model_feature.classifier = new_classifier   # make a new model based on new classifier
            return model_feature 
    
        if self.model_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            last_layer_removed = list(model.children())[:-1]   # all layers except last layer
            return torch.nn.Sequential(*last_layer_removed)    # make a new model based on last_layer_removed
    
        '''
        if self.model_name in ['inception_v3']:
            last_layer_removed = list(model.children())[:-1]   # all layers except last layer
            return torch.nn.Sequential(*last_layer_removed)    # make a new model based on last_layer_removed
        '''
        
        if self.model_name in ['googlenet']:
            last_layer_removed = list(model.children())[:-1]   # all layers except last layer
            return torch.nn.Sequential(*last_layer_removed)    # make a new model based on last_layer_removed
    
        if self.model_name in ['mobilenet_v2']:
            model_feature = copy.deepcopy(model)
            last_layer_removed = list(model_feature.classifier.children())[:-1]   # all layers except last layer
            new_classifier = nn.Sequential(*last_layer_removed)
            model_feature.classifier = new_classifier   # make a new model based on new classifier
            return model_feature  
        
        if self.model_name in ['mnasnet0_5', 'mnasnet1_0']:
            model_feature = copy.deepcopy(model)
            last_layer_removed = list(model_feature.classifier.children())[:-1]   # all layers except last layer
            new_classifier = nn.Sequential(*last_layer_removed)
            model_feature.classifier = new_classifier   # make a new model based on new classifier
            return model_feature  


    def build_model_feature_pooling(self, model):
        '''
        Build a model for feature extraction based on the structure of pre-trained model
        The last AdaptiveAvgPool2d and the last linear layer will be removed.
        Feature extraction before AdaptiveAvgPool2d. 
        '''
        if self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
            pooling_layer_removed = list(model.children())[:-2]   # all layers except last layer and pooling layer
            return torch.nn.Sequential(*pooling_layer_removed)    # make a new model based on pooling_layer_removed
    
        elif self.model_name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            pooling_layer_removed = list(model.children())[:-2]   # all layers except last layer and pooling layer
            return torch.nn.Sequential(*pooling_layer_removed)    # make a new model based on pooling_layer_removed
    
        elif self.model_name in ['alexnet']:
            pooling_layer_removed = list(model.children())[:-2]   # all layers except last layer and pooling layer
            return torch.nn.Sequential(*pooling_layer_removed)    # make a new model based on pooling_layer_removed
    
        elif self.model_name in ['googlenet']:
            pooling_layer_removed = list(model.children())[:-3]   # all layers except last layer and pooling layer
            return torch.nn.Sequential(*pooling_layer_removed)    # make a new model based on pooling_layer_removed
    
        else:
            raise Exception('model_name is not valid to extract features before Average Pooling layer')
   

    def load_model(self, model_path=None):
        '''
        Load a pre-trained model on CPU
        By default, model is loaded on training status (model.training = True) 
        '''
        
        if model_path is None:
            model_path = os.path.join(self.result_main_path, '{}-bestmodel.pth'.format(self.model_name)).replace('\\', '/')

        # Loading model state dictionary from pre-trained model
        # After training, only state dictionary is saved, not the whole model
        # It is recommended to load on CPU to avoid unpredictable error. Later on, you can put model on GPU
        model_state_dict = torch.load(model_path, map_location="cpu")
        
        # When using data parallel during training, it will add "module." phrase to the name of state_dict Keys.
        # If you want to predict model on CPU or single GPU, you need to remove "module." phrase.
        # create new OrderedDict that does not contain "module." phrase in the dictionary Key names.
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k[:7] == 'module.':   # The first 7 characters of state_dict Key name is "module."
                name = k[7:]   # remove `module.` from state_dict Key name
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        
        
        # create a model with random W, bias ; model type should be similar to the trained model
        model = self.build_model(False)   
    
        model.load_state_dict(new_state_dict)   # load W, bias from trained model
        
        # If you want to put model on Cuda (to use GPU), you need to put all inputs to the model on Cuda like "img = img.cuda()". 
        # Also, you need to put model_feature on Cuda too. Check if model is on Cuda : "next(model.parameters()).is_cuda"
        '''
        if cuda:   
            model.cuda()
        '''
             
        if model.training:   # if model is on training, every time you run model(inputs), you get different result 
            model.eval()   # change model status into evaluation, check model status with "model.training" attribute
        
        return model   # model.training is False, model in not on Cuda


    def extract_features_single_image(self, image_path, model, model_feature):
        '''
        Extract features for a single image
        model_feature: model without the last linear layer to extract feature
        '''
    
        img = Image.open(image_path)
        img_trans = self.val_transforms(img).reshape(1, self.channel_num, self.input_img_size[0], self.input_img_size[0])
        '''
        If you don't want to use channel_num here, use following code:
        img_trans = val_transforms(img)
        img_trans = img_trans.reshape(1, img_trans.shape[0], input_img_size[0], input_img_size[0])
        '''
            
        # calculate model output and predicted class by the original model
        output = model(img_trans)
        predicted = torch.argmax(output).item()
        
        # calculate features
        x = model_feature(img_trans)   # torch tensor , dtype=float32
        
        # print('Feature tensor size : ', x.shape)
        
        x = x.view(x.size(0), -1)   # torch tensor , resize to [1, No. of features]
        x = x.detach().numpy()   # Numpy array , dtype=float32
        
        return x, predicted   # feature, predicted Class
        
                
    def extract_features(self, model, model_feature_list, save_summary=True):
        '''
        Get several pre-trained model, extract features for a dataset, similar to extract_features but it accepts several models
        Save results in Excel files in folder "features"
        '''
        
        print('\n feature extraction was called \n')
    
        features_dir = os.path.join(self.result_main_path, "features").replace('\\', '/')
        if not os.path.exists(features_dir):
            os.mkdir(features_dir)
        
        # train dataset
        data = []   # [image_filename, train or val, class_name, predicted]
        for image_filename, class_name, image_path in self.img_loader(self.train_path):                
            print('filename : ', image_filename, ', class name : ', class_name)
            feature = np.empty((0,0), dtype=np.float32)
            for model_feature in model_feature_list:
                x, predicted = self.extract_features_single_image(image_path, model, model_feature)
                feature = np.append(feature, x)
            feature = feature.reshape(1, -1)
            data.append( [image_filename, "train", class_name, self.class_names[predicted]] )
            save_csv(os.path.join(features_dir, "features_train.csv").replace('\\', '/'), feature)
    
        # val dataset
        for image_filename, class_name, image_path in self.img_loader(self.val_path):
            print('filename : ', image_filename, ', class name : ', class_name)
            feature = np.empty((0,0), dtype=np.float32)
            for model_feature in model_feature_list:
                x, predicted = self.extract_features_single_image(image_path, model, model_feature)
                feature = np.append(feature, x)
            feature = feature.reshape(1, -1)
        data.append( [image_filename, "val", class_name, self.class_names[predicted]] )
        save_csv(os.path.join(features_dir, "features_val.csv").replace('\\', '/'), feature)
    
        # Save summary
        if save_summary:
            writer = pd.ExcelWriter(os.path.join(features_dir, "features_summary.xlsx").replace('\\', '/'))
            column_name = ['Filename', 'train/val', 'Class', 'Prediction']
            data_df = pd.DataFrame(data, columns = column_name)
            data_df.to_excel(writer, "result", index=False)
            writer.save()         


    def img_loader(self, DatasetPath):   # files should be stored in sub-folders inside DatasetPath
        
        for root, directories, _ in os.walk(DatasetPath):
            for sub_folder in directories:
                for _, _, filenames in os.walk(os.path.join(DatasetPath, sub_folder)):
                    for filename in filenames:
                        yield filename, sub_folder, os.path.join(DatasetPath, sub_folder, filename).replace('\\', '/')   # image filename, subfolder name, image path
            break
    
    
    def img_loader_f(self, DatasetPath):   # input images should be stored in one folder (Dataset)
        for root, _, filenames in os.walk(DatasetPath):
            for filename in filenames:
                yield filename, os.path.join(DatasetPath, filename).replace('\\', '/')   # image filename, image path      
            break


class Predictor():
    
    def __init__(self, model_name, classes_num, input_img_size, channel_num, result_folder,
                 batch_size, mean, std, test_transforms):
        
        self.model_name = model_name
        self.classes_num = classes_num
        self.input_img_size = input_img_size
        self.channel_num = channel_num
        self.result_folder = result_folder
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.test_transforms = test_transforms

        model_name_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'] + \
                  ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'] + \
                  ['alexnet'] + ['densenet121', 'densenet161', 'densenet169', 'densenet201'] + ['googlenet'] + ['mobilenet_v2'] + ['mnasnet0_5', 'mnasnet1_0']

        if self.model_name not in model_name_list:
            raise Exception('model_name is not valid')
        
        self.GPU_num = torch.cuda.device_count()

        self.cuda = torch.cuda.is_available()

        self.result_main_path = "./trained_models/{}".format(self.result_folder)
        if not os.path.exists(self.result_main_path):
            os.mkdir(self.result_main_path)  
        

    def build_model(self, ImageNet_pretrained):
        '''
        By default, model is built on training status (model.training = True)
        '''
        
        # old method, this methos only works for one model. 
    #    if self.model_name == 'resnet18':
    #        if not ImageNet_pretrained:
    #            return models.resnet18(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
    #        else:
    #            model = models.resnet18(pretrained=ImageNet_pretrained, num_classes=1000)
    #            num_ftrs = model.fc.in_features   # No. of input features to fully connected layer
    #            model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
    #            return model
        
        if self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.fc.in_features   # No. of input features to fully connected layer
                model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[6].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[6] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['alexnet']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[6].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[6] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier.in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        '''
        if self.model_name in ['inception_v3']:
            # This network is unique because it has two output layers when training. The second output is an auxiliary output. When testing, we only consider the primary output.
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.fc.in_features   # No. of input features to fully connected layer, last linear layer
                num_ftrs_aux = model.AuxLogits.fc.in_features   # auxiliary output
                model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                model.AuxLogits.fc = nn.Linear(num_ftrs_aux, self.classes_num)   # auxiliary output
                return model
        '''
        
        if self.model_name in ['googlenet']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.fc.in_features   # No. of input features to fully connected layer, last linear layer
                model.fc = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['mobilenet_v2']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[1].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[1] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    
        if self.model_name in ['mnasnet0_5', 'mnasnet1_0']:
            model_object = getattr(models, self.model_name)
            if not ImageNet_pretrained:
                return model_object(pretrained=ImageNet_pretrained, num_classes=self.classes_num)
            else:
                model = model_object(pretrained=ImageNet_pretrained, num_classes=1000)
                num_ftrs = model.classifier[1].in_features   # No. of input features to fully connected layer, last linear layer
                model.classifier[1] = nn.Linear(num_ftrs, self.classes_num)   # modify fc layer , No. of outputs is set to self.classes_num
                return model
    

    def load_model(self, model_path=None):
        '''
        Load a pre-trained model on CPU
        By default, model is loaded on training status (model.training = True) 
        '''
        
        if model_path is None:
            model_path = os.path.join(self.result_main_path, '{}-bestmodel.pth'.format(self.model_name)).replace('\\', '/')
        print(model_path)
        # Loading model state dictionary from pre-trained model
        # After training, only state dictionary is saved, not the whole model
        # It is recommended to load on CPU to avoid unpredictable error. Later on, you can put model on GPU
        model_state_dict = torch.load(model_path, map_location="cpu")
        
        # When using data parallel during training, it will add "module." phrase to the name of state_dict Keys.
        # If you want to predict model on CPU or single GPU, you need to remove "module." phrase.
        # create new OrderedDict that does not contain "module." phrase in the dictionary Key names.
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k[:7] == 'module.':   # The first 7 characters of state_dict Key name is "module."
                name = k[7:]   # remove `module.` from state_dict Key name
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        
        
        # create a model with random W, bias ; model type should be similar to the trained model
        model = self.build_model(False)   
    
        model.load_state_dict(new_state_dict)   # load W, bias from trained model
        
        # If you want to put model on Cuda (to use GPU), you need to put all inputs to the model on Cuda like "img = img.cuda()". 
        # Also, you need to put model_feature on Cuda too. Check if model is on Cuda : "next(model.parameters()).is_cuda"
        '''
        if cuda:   
            model.cuda()
        '''
             
        if model.training:   # if model is on training, every time you run model(inputs), you get different result 
            model.eval()   # change model status into evaluation, check model status with "model.training" attribute
        
        return model   # model.training is False, model in not on Cuda


    def predict_single_image(self, image_path, model):
        '''
        Predict Class or Label of a single image
        '''
        model.eval()
    
        img = Image.open(image_path)
        img_trans = self.test_transforms(img).reshape(1, self.channel_num, self.input_img_size[0], self.input_img_size[0])
        '''
        If you don't want to use channel_num here, use following code:
        img_trans = test_transforms(img)
        img_trans = img_trans.reshape(1, img_trans.shape[0], input_img_size[0], input_img_size[0])
        '''
            
        # calculate model output and predicted class by the original model
        output = model(img_trans)
    
        predicted = torch.argmax(output).item()
        
        print('predicted class index : ', predicted, '\n')
        
        return output, predicted


    def predict_images(self, image_folder, model):
        '''
        Predict Class or Label of several images
        '''
        output_all, predicted_all = [], []
        # Load images, call "predict_single_image"
        for image_filename, image_path in self.img_loader_f(image_folder):
                print('filename in process : ', image_filename)
                output, predicted = self.predict_single_image(image_path, model)
                output_all.append(output)
                predicted_all.append(predicted)
        
        return output_all, predicted_all


    def predict_dataset(self, dataset_path, model, criterion, plot_cm=False):
        '''
        Predict Class or Label of a dataset
        '''
    
        dataset_folders = datasets.ImageFolder(dataset_path, self.test_transforms)
        
        dataset_loader = torch.utils.data.DataLoader(dataset_folders, batch_size=self.batch_size, shuffle=False)
        
        # check data size
        dataset_iter = iter(dataset_loader)
        inputs, labels = next(dataset_iter)
        print('\ninput tensor size : ', inputs.shape, '\nlabel tensor size : ', labels.shape, '\n')
        
        dataset_img_num = len(dataset_folders)     # No. of images
        classes_num = len(dataset_folders.classes)
        class_names = dataset_folders.classes  # name of classes
    
        print('No. of images in dataset : ', dataset_img_num, ',  No. of classes :', classes_num, '\n')
    
    
        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_img = 0.0
        predicted_all = np.empty((0, 1))
        label_all = np.empty((0, 1))
    
        with torch.no_grad():   # temporarily set all the requires_grad flags to false (no gradient calculations).
            
            for batch_index, (img, label) in enumerate(dataset_loader):
                
                if self.cuda:
                    img, label = img.cuda(), label.cuda()
                    model = model.cuda()
                img.requires_grad = False
                label.requires_grad = False
                
                output = model(img)
              
                loss = criterion(output, label)
                _, predicted = torch.max(output.data, 1)
                total_loss += loss.item()
                total_correct += (predicted == label.data).sum().item()
                total_img += label.size(0)   # No. of images in the epoch or batch_size
    
                predicted_all = np.append(predicted_all, predicted.cpu().data.numpy())   # put Tensor on CPU, then convert to numpy array
                label_all = np.append(label_all, label.cpu().data.numpy())
                
                print(" BATCH [{}] ACC [{:.4}] loss [{:.4}]".format(batch_index, 
                      (predicted == label.data).sum().item() / label.size(0), loss.item()))
    
        dataset_acc = total_correct / total_img
        dataset_loss = total_loss / total_img
    
        print(" Dataset : ACC [{:.4}] loss [{:.4}]".format(dataset_acc, dataset_loss))
    
        # confusion matrix
        cm = confusion_matrix(label_all.tolist(), predicted_all.tolist())
        
        pred_labels = ['Predicted '+ l for l in class_names]
        cm_df = pd.DataFrame(cm, index=class_names, columns=pred_labels)
        print("\n Confusion Matrix : \n", cm_df, "\n")
        
        model = model.cpu()   # some functions require model on CPU. 
        
        visualize_prediction(dataset_path, os.path.join(self.result_main_path, "Prediction_Result.png").replace('\\', '/'),
                     model, self.model_name, self.test_transforms, self.mean, self.std, self.cuda, images_num=12, row=2, title = 'Model Prediction')
        
        if plot_cm:
            plot_confusion_matrix(cm, class_names, os.path.join(self.result_main_path, "Prediction_cm.png").replace('\\', '/'), normalize=True)

        
        return dataset_acc, dataset_loss, label_all, predicted_all, cm, cm_df


    def img_loader_f(self, DatasetPath):   # input images should be stored in one folder (Dataset)
        for root, _, filenames in os.walk(DatasetPath):
            for filename in filenames:
                yield filename, os.path.join(DatasetPath, filename).replace('\\', '/')   # image filename, image path      
            break
