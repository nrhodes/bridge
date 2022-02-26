# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from typing import Any
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models import ResNet

LETTERS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


def PbnDealToTensor(deal):
  ''' Returns 4x4x13 tensor containing the deal.
  First dimension is the direction: W is always first, then N, then E, then S
  '''
  def letterToIndex(c):
    return LETTERS.index(c)

  hands = torch.zeros((4, 4, 13 ))
  first = deal[0]
  positions = ['W', 'N', 'E', 'S']
  firstIndex = positions.index(first)
  for handIndex, hand in enumerate(deal[2:].split(' ')):
    for suitIndex, suit in enumerate(hand.split('.')):
      for card in suit:
        hands[(handIndex + firstIndex) % 4, suitIndex, letterToIndex(card)] = 1
  return hands;

deal = PbnDealToTensor("W:...2 ...3 ...4 ...5")

def AddTrumpToDeal(dealTensor, trumpSuit):
  ''' Returns 5x4x13 tensor containing the deal incl. trump (trump is one of
  'S', 'H', 'D', 'C', 'N')
  First dimension trump: 1x13 all ones if trump.  4x13 all zeros if notrump
  '''
  trumpTensor = torch.zeros((1, 4, 13 ))
  if trumpSuit != 'N':
    trumps = ['S', 'H', 'D', 'C']
    trumpIndex = trumps.index(trumpSuit);
    trumpTensor[0, trumpIndex] = torch.ones((1, 13))
  return torch.cat([trumpTensor, dealTensor])

from fastai.vision.all import *

import pandas as pd

def TensorToPbn(t):
  def HandToPbn(hand):
    def HandSuitToPbn(suit):
      mapping = ['2']
      pips = [LETTERS[i] if suit[i] != 0 else ' ' for i in range(13)]
      f = filter(lambda c: c != ' ', pips)
      return ' '.join(f)

    suits = [HandSuitToPbn(hand[suit]) for suit in range(4)]
    return '.'.join(suits)

  hands = [HandToPbn(t[hand]) for hand in range(4)]
  return "W:" + ' '.join(hands)


print(TensorToPbn(deal))

class TensorBridgeHand(TensorBase):   pass

class BridgeHandSetup(Transform):
    "Transform that converts from bridge hands to tensor"

    def encodes(self, o): return TensorBridgeHand(AddTrumpToDeal(PbnDealToTensor(o), 'N'))
    def decodes(self, o): return TitledStr(TensorToPbn(o))


def BridgeHandBlock():
    "`TransformBlock` for bridge hands"
    return TransformBlock(type_tfms=BridgeHandSetup())

BRIDGE_HANDS=np.array(["W:...2 ...3 ...4 ...5#S"]*50)

x=BridgeHandBlock()
x.__dict__

dblock = DataBlock(blocks=[BridgeHandBlock, CategoryBlock],
                   get_x=lambda d: d.split('#')[0],
                   get_y=lambda d: d.split('#')[1],
                   splitter=RandomSplitter())
dsets=dblock.datasets(BRIDGE_HANDS)

dblock.summary(BRIDGE_HANDS)

dsets = dblock.datasets(BRIDGE_HANDS, verbose=True)
# print(dsets.subset(0))
# pdb.set_trace()
# dls = dsets.dataloaders(path='.', after_item=dblock.item_tfms, after_batch=dblock.batch_tfms, verbose=True)
# train=dls.train
# print(len(train))
# train.show_batch()
dls = dsets.dataloaders(bs=10);
train = dls.train
train.show_batch()
dls.show_batch()
from typing import Type, Any, Callable, Union, List, Optional


class ResNetCustom(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def custom_resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNetCustom(BasicBlock, [5, 4, 6, 3], **kwargs)
    return model


learn = cnn_learner(dls, custom_resnet34,
                    pretrained=False,
                    metrics=accuracy,
                    cbs=[#SaveModelCallback(),
                         #EarlyStoppingCallback(monitor='valid_loss', min_delta=0.02, patience=30),
                        # ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=10)
                         ])
learn.fit(5)
