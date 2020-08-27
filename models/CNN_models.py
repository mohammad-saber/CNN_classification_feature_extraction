
import torch.nn as nn


# CNN with 1 Conv layer , no batch normalization , max Pooling
class CNN_1L(nn.Module):

    def __init__(self, channel_num, input_img_size, classes_num):

        super().__init__()

        # Convolution 1
        # in_channels= 1 or 3
        # out_channels=16 : No. of kernels = No. of feature maps = No. of outputs
        # kernel_size=5 : 5*5
        # padding = (k-1)/2
        self.conv1 = nn.Conv2d(in_channels=channel_num, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.activation_fcn1 = nn.ReLU()    # after each convolution, there should be a non-linear function

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        # classes_num : No. of outputs = No. of labels
        self.fc1 = nn.Linear(16 * input_img_size//2 * input_img_size//2, classes_num)


    def forward(self, x):

        # Convolution 1
        out = self.conv1(x)
        out = self.activation_fcn1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Resize is necessary for linear function (flattening), CNN doesn`t need flattening of input data.
        # Original size: (batch_size, out_channels, H, W)
        # New out size: (batch_size, out_channels*H*W)
        # out.size(0) = batch_size
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


# CNN with 1 Conv layer , batch normalization , max Pooling
class CNN_1L_bn(nn.Module):

    def __init__(self, channel_num, input_img_size, classes_num):

        super().__init__()

        # Convolution 1
        # in_channels= 1 or 3
        # out_channels=16 : No. of kernels = No. of feature maps = No. of outputs
        # kernel_size=5 : 5*5
        # padding = (k-1)/2
        self.conv1 = nn.Conv2d(in_channels=channel_num, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation_fcn1 = nn.ReLU()    # after each convolution, there should be a non-linear function

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        # classes_num : No. of outputs = No. of labels
        self.fc1 = nn.Linear(16 * input_img_size//2 * input_img_size//2, classes_num)


    def forward(self, x):

        # Convolution 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fcn1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Resize is necessary for linear function (flattening), CNN doesn`t need flattening of input data.
        # Original size: (batch_size, out_channels, H, W)
        # New out size: (batch_size, out_channels*H*W)
        # out.size(0) = batch_size
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


# CNN with 2 Conv layer , no batch normalization , max Pooling
class CNN_2L(nn.Module):

    def __init__(self, channel_num, input_img_size, classes_num):
        super().__init__()

        # Convolution 1
        # in_channels= 1 or 3
        # out_channels=16 : No. of kernels = No. of feature maps = No. of outputs
        # kernel_size=5 : 5*5
        # padding = (k-1)/2
        self.conv1 = nn.Conv2d(in_channels=channel_num, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.activation_fcn1 = nn.ReLU()  # after each convolution, there should be a non-linear function

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.activation_fcn2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        # classes_num : No. of outputs = No. of labels
        self.fc1 = nn.Linear(32 * input_img_size//4 * input_img_size//4, classes_num)

    def forward(self, x):
        # Convolution 1
        out = self.conv1(x)
        out = self.activation_fcn1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.activation_fcn2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize is necessary for linear function (flattening), CNN doesn`t need flattening of input data.
        # Original size: (batch_size, out_channels, H, W)
        # New out size: (batch_size, out_channels*H*W)
        # out.size(0) = batch_size
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


# CNN with 2 Conv layer , batch normalization , max Pooling
class CNN_2L_nb(nn.Module):

    def __init__(self, channel_num, input_img_size, classes_num):
        super().__init__()

        # Convolution 1
        # in_channels= 1 or 3
        # out_channels=16 : No. of kernels = No. of feature maps = No. of outputs
        # kernel_size=5 : 5*5
        # padding = (k-1)/2
        self.conv1 = nn.Conv2d(in_channels=channel_num, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation_fcn1 = nn.ReLU()  # after each convolution, there should be a non-linear function

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.activation_fcn2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        # classes_num : No. of outputs = No. of labels
        self.fc1 = nn.Linear(32 * input_img_size//4 * input_img_size//4, classes_num)

    def forward(self, x):
        # Convolution 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fcn1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fcn2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize is necessary for linear function (flattening), CNN doesn`t need flattening of input data.
        # Original size: (batch_size, out_channels, H, W)
        # New out size: (batch_size, out_channels*H*W)
        # out.size(0) = batch_size
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


