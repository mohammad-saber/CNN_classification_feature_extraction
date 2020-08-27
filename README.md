# CVTeam_CNN_classification_feature_extraction
This repository is the implementation of CNN for classification and feature extraction in pytorch. Pytorch pretrained models have been used which are explained [here](https://pytorch.org/docs/stable/torchvision/models.html).

This code supports data parallelism and multipl GPU. Also, you can select to load pretrained weights (trained on ImageNet dataset) or train from scratch using random weights. 

Pretrained model structure has 1000 nodes in the last layer. This code modifies the last layer of all models to be campatible with every dataset. 

Following models can be used:
```
'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
'alexnet' + 'densenet121', 'densenet161', 'densenet169', 'densenet201',
'googlenet' + 'mobilenet_v2' + 'mnasnet0_5', 'mnasnet1_0'
```

## Main Requirements

* Python 3.7.6

* Pytorch 1.4.0


## Install

```
git clone CVTeam_CNN_classification_feature_extraction
cd CVTeam_CNN_classification_feature_extraction
pip install -r requirements.txt
```
If conditions are not met, please install pytorch based on the instructions in [pytorch documentation](https://pytorch.org/).

## Code Structure

In `train.py` script, you can enter parameters, and make instance of the `Trainer class`. This class contains all methods to initialize the model, train, and extract features. Following is descrption of main methods of Trainer class:

- load_dataset: load dataset using pytorch dataloader
- train: train the model 
- build_model: build model for training
- build_model_feature: build a new model for feature extraction before last linear layer
- build_model_feature_pooling: build a new model for feature extraction before adaptive average pooing layer (models without adaptive average pooing layer are excluded here which are: densenet, mobilenet, mnasnet)
- extract_features_single_image: extract features for a single image except for VGG16 intermediate layer feature extraction
- extract_features: extract features of a dataset. 

## Dataset structure

Prepare data in the following format:

```text
{DATA_HOME}
    ├─ train
    │    ├─ class A
    │    ├─ class B
    │    └─ ...
    └─ val
         ├─ class A
         ├─ class B
         └─ ...
```


### How to use
* **Train**:

In `train.py` script, enter the desired parameters. Then, run: 

```
python train.py
```


* **Prediction**:

In `prediction.py` script, enter the desired parameters (some parameters are similar to parameters in `train.py`). Then, run: 

```
python prediction.py
```


## Pytorch models download error

When downloading pytorch models, If the following error happens:

```
ImportError: FloatProgress not found. Please update jupyter and ipywidgets. 

See https://ipywidgets.readthedocs.io/en/stable/user_install.html
```

To solve this issue, run the following commands in terminal:

```
conda install -c conda-forge ipywidgets

jupyter nbextension enable --py widgetsnbextension
```

If you have installed pytorch in a virtual environment, run the above commands after activating the virtual environment.


