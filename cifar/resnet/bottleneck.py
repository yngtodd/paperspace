'''ResNet in PyTorch.

Branch: adapt

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
    short_planes = 1

    def __init__(self, in_planes, planes1,
                 planes2, planes3,
                 kernel1=1, dropout1=0.0,
                 kernel2=3, dropout2=0.0,
                 kernel3=1, dropout3=0.0,
                 shortcut_kernel=1, stride=1):
        super(Bottleneck, self).__init__()
        #self.short_planes = short_planes
        self.planes1 = planes1
        self.planes2 = planes2
        self.planes3 = planes3
        print(self.planes3)
        self.conv1 = nn.Conv2d(in_planes, self.planes1, kernel_size=kernel1, bias=False)
        self.drop1 = nn.Dropout2d(p=dropout1)
        self.bn1 = nn.BatchNorm2d(self.planes1)
        self.conv2 = nn.Conv2d(self.planes1, self.planes2, kernel_size=kernel2, stride=stride, padding=1, bias=False)
        self.drop2 = nn.Dropout2d(p=dropout2)
        self.bn2 = nn.BatchNorm2d(self.planes2)
        self.conv3 = nn.Conv2d(self.planes2, self.expansion*self.planes3, kernel_size=kernel3, bias=False)
        self.drop3 = nn.Dropout2d(p=dropout3)
        self.bn3 = nn.BatchNorm2d(self.expansion*self.planes3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*self.planes3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*self.planes3, kernel_size=shortcut_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*self.planes3)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.drop1(self.conv1(x))))
        out = F.relu(self.bn2(self.drop2(self.conv2(out))))
        #print('out at conv2 has size {}'.format(out.size()))
        out = self.bn3(self.drop3(self.conv3(out)))
        print('out has size {}'.format(out.size()))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_kernel=3,
                 l1kernel1=1, l1kernel2=3, l1kernel3=1, dropout1=0.0, stride1=1,
                 l2kernel1=1, l2kernel2=3, l2kernel3=1, dropout2=0.0, stride2=2,
                 l3kernel1=1, l3kernel2=3, l3kernel3=1, dropout3=0.0, stride3=2,
                 l4kernel1=1, l4kernel2=3, l4kernel3=1, dropout4=0.0, stride4=2,
                 num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.initial_kernel = initial_kernel
        # Block 1
        self.l1kernel1 = l1kernel1
        self.l1kernel2 = l1kernel2
        self.l1kernel3 = l1kernel3
        self.dropout1 = dropout1
        self.stride1 = stride1
        # Block 2
        self.l2kernel1 = l2kernel1
        self.l2kernel2 = l2kernel2
        self.l2kernel3 = l2kernel3
        self.dropout2 = dropout2
        self.stride2 = stride2
        # Block 3
        self.l3kernel1 = l3kernel1
        self.l3kernel2 = l3kernel2
        self.l3kernel3 = l3kernel3
        self.dropout3 = dropout3
        self.stride3 = stride3
        # Block 4
        self.l4kernel1 = l4kernel1
        self.l4kernel2 = l4kernel2
        self.l4kernel3 = l4kernel3
        self.dropout4 = dropout4
        self.stride4 = stride4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=self.initial_kernel, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes1, planes2, planes3, num_blocks, kernel1, dropout1,
                    kernel2, dropout2, kernel3, dropout3, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in strides:
            #print('planes is {}'.format(planes[i]))
            layers.append(
                block(
                    self.in_planes, planes1=planes1,
                    kernel1=kernel1, dropout1=dropout1,
                    planes2=planes2,
                    kernel2=kernel2, dropout2=dropout2,
                    planes3=planes3,
                    kernel3=kernel3, dropout3=dropout3, stride=stride
                )
            )
            self.in_planes = planes3 * block.expansion

        return nn.Sequential(*layers)

    def _calculate_planes(self, input_size, kernel, padding, stride):
        """Calculate the output size of a convolution layer."""
        return ((input_size + 2*padding - kernel) // stride) + 1

    def _make_layer1(self):
        #print(self.in_planes, self.l1kernel1, self.stride1)
        l1planes1 = self._calculate_planes(input_size=self.in_planes, kernel=self.l1kernel1, padding=0, stride=self.stride1)
        l1planes2 = self._calculate_planes(input_size=l1planes1, kernel=self.l1kernel2, padding=1, stride=self.stride1)
        #print('input for l1planes3 - input: {}, kernel: {}, stride: {}'.format(l1planes2, self.l1kernel3, self.stride1))
        l1planes3 = self._calculate_planes(input_size=l1planes2, kernel=self.l1kernel3, padding=0, stride=self.stride1)
        planes = [l1planes1, l1planes2, l1planes3]
        #self.l1short_planes3 = self._calculate_planes(input_size=self.in_planes, kernel=self.shortcut_kernel, padding=0, stride=self.stride1)

        return self._make_layer(
                self.block, planes1=l1planes1, num_blocks=self.num_blocks[0],
                kernel1=self.l1kernel1, dropout1=self.dropout1,
                planes2=l1planes2,
                kernel2=self.l1kernel2, dropout2=self.dropout2,
                planes3=l1planes3,
                kernel3=self.l1kernel3, dropout3=self.dropout3, stride=self.stride1
            )

    def _make_layer2(self):
        l2planes1 = self._calculate_planes(self.in_planes, self.l2kernel1, padding=0, stride=self.stride2)
        l2planes2 = self._calculate_planes(l2planes1, self.l2kernel2, padding=1, stride=self.stride2)
        l2planes3 = self._calculate_planes(l2planes2, self.l2kernel3, padding=0, stride=self.stride2)
        planes = [l2planes1, l2planes2, l2planes3]
        #self.l2short_planes3 = self._calculate_planes(self.in_planes, self.shortcut_kernel, padding=0, stride=self.stride2)

        return self._make_layer(
                self.block, planes1=l2planes1, num_blocks=self.num_blocks[1],
                kernel1=self.l2kernel1, dropout1=self.dropout1,
                planes2=l2planes2,
                kernel2=self.l2kernel2, dropout2=self.dropout2,
                planes3=l2planes3,
                kernel3=self.l2kernel3, dropout3=self.dropout3, stride=self.stride2
            )

    def _make_layer3(self):
        l3planes1 = self._calculate_planes(self.in_planes, self.l3kernel1, padding=0, stride=self.stride3)
        l3planes2 = self._calculate_planes(l3planes1, self.l3kernel2, padding=1, stride=self.stride3)
        l3planes3 = self._calculate_planes(l3planes2, self.l3kernel3, padding=0, stride=self.stride3)
        planes = [l3planes1, l3planes2, l3planes3]
        #self.l3short_planes3 = self._calculate_planes(self.in_planes, self.shortcut_kernel, padding=0, stride=self.stride3)

        return self._make_layer(
                self.block, planes1=l3planes1, num_blocks=self.num_blocks[2],
                kernel1=self.l3kernel1, dropout1=self.dropout1,
                planes2=l3planes2,
                kernel2=self.l3kernel2, dropout2=self.dropout2,
                planes3=l3planes3,
                kernel3=self.l3kernel3, dropout3=self.dropout3, stride=self.stride3
            )

    def _make_layer4(self):
        l4planes1 = self._calculate_planes(self.in_planes, self.l4kernel1, padding=0, stride=self.stride4)
        l4planes2 = self._calculate_planes(l4planes1, self.l4kernel2, padding=1, stride=self.stride4)
        l4planes3 = self._calculate_planes(l4planes2, self.l4kernel3, padding=0, stride=self.stride4)
        #l4short_planes3 = self._calculate_planes(self.in_planes, self.shortcut_kernel, padding=0, stride=self.stride4)
        planes = [l4planes1, l4planes2, l4planes3]

        return self._make_layer(
                self.block, planes1=l4planes1, num_blocks=self.num_blocks[3],
                kernel1=self.l4kernel1, dropout1=self.dropout1,
                planes2=l4planes2,
                kernel2=self.l4kernel2, dropout2=self.dropout2,
                planes3=l4planes3,
                kernel3=self.l4kernel3, dropout3=self.dropout3, stride=self.stride4
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
        out = F.adaptive_avg_pool2d(out, 4)
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


def test(verbose=True):
    model = ResNet50()

    if verbose:
        print('Named model parameters that require gradients:\n')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    y = model(Variable(torch.randn(1,3,32,32)))
    print('\nModel output has shape {}'.format(y.size()))

test(verbose=False)
