'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from inspect import signature
from collections import defaultdict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel1=1, dropout1=0.0,
                 kernel2=3, dropout2=0.0, kernel3=1, dropout3=0.0,
                 shortcut_kernel=1, stride=1):
        super(BasicBlock, self).__init__()
        self.shortcut_kernel = shortcut_kernel
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel1, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel2, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=self.shortcut_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, kernel1=1, dropout1=0.0,
                 kernel2=3, dropout2=0.0, kernel3=1, dropout3=0.0,
                 shortcut_kernel=1, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel1, bias=False)
        self.drop1 = nn.Dropout2d(p=dropout1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel2, stride=stride, padding=1, bias=False)
        self.drop2 = nn.Dropout2d(p=dropout2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=kernel3, bias=False)
        self.drop3 = nn.Dropout2d(p=dropout3)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=shortcut_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.drop1(self.conv1(x))))
        out = F.relu(self.bn2(self.drop2(self.conv2(out))))
        out = self.bn3(self.drop3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_kernel=3,
                 planes1=64, l1kernel1=1, l1kernel2=3, l1kernel3=1, dropout1=0.0, stride1=1,
                 planes2=128, l2kernel1=1, l2kernel2=3, l2kernel3=1, dropout2=0.0, stride2=2,
                 planes3=256, l3kernel1=1, l3kernel2=3, l3kernel3=1, dropout3=0.0, stride3=2,
                 planes4=512, b4kernel1=1, b4kernel2=3, b4kernel3=1, dropout4=0.0, stride4=2,
                 num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.initial_kernel = initial_kernel
        # Block 1
        self.planes1 = planes1
        self.l1kernel1 = l1kernel1
        self.l1kernel2 = l1kernel2
        self.l1kernel3 = l1kernel3
        self.dropout1 = dropout1
        self.stride1 = stride1
        # Block 2
        self.planes2 = planes2
        self.l2kernel1 = l2kernel1
        self.l2kernel2 = l2kernel2
        self.l2kernel3 = l2kernel3
        self.dropout2 = dropout2
        self.stride2 = stride2
        # Block 3
        self.planes3 = planes3
        self.l3kernel1 = l3kernel1
        self.l3kernel2 = l3kernel2
        self.l3kernel3 = l3kernel3
        self.dropout3 = dropout3
        self.stride3 = stride3
        # Block 4
        self.planes4 = planes4
        self.b4kernel1 = b4kernel1
        self.b4kernel2 = b4kernel2
        self.b4kernel3 = b4kernel3
        self.dropout4 = dropout4
        self.stride4 = stride4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=self.initial_kernel, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, kernel1, dropout1,
                    kernel2, dropout2, kernel3, dropout3, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes,
                    kernel1, dropout1,
                    kernel2, dropout1,
                    kernel3, dropout3, stride=stride
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _calculate_planes(self, input_size, kernel, padding, stride):
        """Calculate the output size of a convolution layer."""
        return ((input_size + 2*padding - kernel) / s) + 1

    def _make_layer1(self):
        return self._make_layer(
                self.block, planes=self.planes1, num_blocks=self.num_blocks[0],
                kernel1=self.l1kernel1, dropout1=self.dropout1,
                kernel2=self.l1kernel2, dropout2=self.dropout2,
                kernel3=self.l1kernel3, dropout3=self.dropout3, stride=self.stride1
            )

    def _make_layer2(self):
        return self._make_layer(
                self.block, planes=self.planes2, num_blocks=self.num_blocks[1],
                kernel1=self.l2kernel1, dropout1=self.dropout1,
                kernel2=self.l2kernel2, dropout2=self.dropout2,
                kernel3=self.l2kernel3, dropout3=self.dropout3, stride=self.stride2
            )

    def _make_layer3(self):
        return self._make_layer(
                self.block, planes=self.planes3, num_blocks=self.num_blocks[2],
                kernel1=self.l3kernel1, dropout1=self.dropout1,
                kernel2=self.l3kernel2, dropout2=self.dropout2,
                kernel3=self.l3kernel3, dropout3=self.dropout3, stride=self.stride3
            )

    def _make_layer4(self):
        return self._make_layer(
                self.block, planes=self.planes4, num_blocks=self.num_blocks[3],
                kernel1=self.b4kernel1, dropout1=self.dropout1,
                kernel2=self.b4kernel2, dropout2=self.dropout2,
                kernel3=self.b4kernel3, dropout3=self.dropout3, stride=self.stride4
            )

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        ----------
        * `deep`: [boolean, optional]
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        -------
        * `params`: [str]
            Mapping of string to any Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def _instantiate_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def set_params(self, **params):
        self._instantiate_params(**params)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = out.view(out.size(), -1)
        out = self.layer1(out)
        out = out.view(out.size(), -1)
        out = self.layer2(out)
        out = out.view(out.size(), -1)
        out = self.layer3(out)
        out = out.view(out.size(), -1)
        out = self.layer4(out)
        out = out.view(out.size(), -1)
        out = F.avg_pool2d(out, 4)
        print('Output of avg pool {}'.format(out.size()))
        out = out.view(out.size(0), -1)
        print('Input to linear layers has shape {}'.format(out.size()))
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test(verbose=False):
    model = ResNet152()

    if verbose:
        print('Named model parameters that require gradients:\n')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    y = model(Variable(torch.randn(1,3,32,32)))
    print('\nModel output has shape {}'.format(y.size()))

test(verbose=False)
