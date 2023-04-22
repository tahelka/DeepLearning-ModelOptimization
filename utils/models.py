import torch
import torch.nn as nn
import torch.nn.functional as F

#from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential
from torch.nn.init import xavier_uniform, xavier_normal


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims, dropouts):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.dropouts = False # not supported in this model type
        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        print(self)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        for x in range(len(self.filters)):
            in_c = self.filters[x-1] if x != 0 else in_channels
            layers.append(nn.Conv2d(in_channels=in_c, out_channels=self.filters[x], kernel_size=3,
                                stride=1, padding=1))
            layers.append(nn.ReLU())
            if (x+1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======
        out_w=in_w
        out_h=in_h
        for i in range(len(self.filters)//self.pool_every):
            out_w = (out_w-2)/2 + 1
            out_h = (out_h - 2) / 2 + 1

        flatSize=int(out_h * out_w * self.filters[-1])
        for x in range(len(self.hidden_dims)):
            in_f = self.hidden_dims[x-1] if x != 0 else flatSize #last conv layer channel output
            layers.append(nn.Linear(in_features=in_f, out_features=self.hidden_dims[x]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),
        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        convOut = self.feature_extractor.forward(x)
        fcInput = torch.flatten(convOut,start_dim=1)
        out = self.classifier.forward(fcInput)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims,  dropouts=True):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims, dropouts)

        self.dropouts = dropouts

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # Implement this function with the fixes you suggested question 1.1. Extra points.
        # ====== YOUR CODE: ======
        for x in range(len(self.filters)):
            in_c = self.filters[x - 1] if x != 0 else in_channels
            layers.append(nn.Conv2d(in_channels=in_c, out_channels=self.filters[x], kernel_size=3,
                                    stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(self.filters[x]))
            if (x + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))
                if self.dropouts:
                    increasedDropout = 0.1 * ((x + 1) // self.pool_every)
                    layers.append(nn.Dropout(p=0.1 + increasedDropout))
        # ========================
        seq = nn.Sequential(*layers)
        return seq
        # ========================
