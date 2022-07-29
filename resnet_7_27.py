import torch.nn as nn
import torch
# from torchsummary import summary
# num_attention_heads=4
# feature=4
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class block(nn.Module):#bottle为sebottle
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2  = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 1)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.relu = nn.ELU()
        self.se = SELayer(planes * 1, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# class block(nn.Module):
#     def __init__(
#             self, in_channels, intermediate_channels, identity_downsample=None, stride=1, _dilation_rate=1):
#         super(block, self).__init__()
#
#         self.conv1 = nn.Conv2d(
#             intermediate_channels, intermediate_channels, kernel_size=3, bias=False, padding=2 * _dilation_rate,dilation =_dilation_rate )
#         self.ins1 = nn.BatchNorm2d(intermediate_channels)
#         self.elu = nn.ELU()
#         self.dropout1 = nn.Dropout(0.15)
#         self.conv2 = nn.Conv2d(
#             intermediate_channels, intermediate_channels, kernel_size=3, bias=False,dilation =_dilation_rate  )
#         self.ins2 = nn.BatchNorm2d(intermediate_channels)
#
#     def forward(self, x):
#         identity = x.clone()
#         x = self.conv1(x)
#         x = self.ins1(x)
#         x = self.elu(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = self.ins2(x)
#         x += identity
#         x = self.elu(x)
#         return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.filter = 64
        self.in_channels = image_channels
        # self.conv1 = nn.Conv2d(image_channels, self.filter, kernel_size=1, stride=1, padding=0, bias=False)
        # # self.ins = nn.BatchNorm2d(self.filter)
        # self.ins = nn.InstanceNorm2d(self.filter)
        # self.elu = nn.ELU()
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=self.filter, stride=1
        )
        # self.last_layer = nn.Conv2d(self.filter, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.last_layer = nn.Conv2d(self.filter, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.ins(x)
        # x = self.elu(x)
        x = self.layer1(x)
        # x = self.last_layer(x)
        # x = self.sigmoid(x)
        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):

        layers = []

        dilation_rate = 1
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.filter, intermediate_channels))
            # dilation_rate = dilation_rate * 2
            # if dilation_rate >16:
            #     dilation_rate = 1
        return nn.Sequential(*layers)
