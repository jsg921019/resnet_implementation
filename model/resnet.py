import torch
import torch.nn as nn

def residual_function(in_channels, out_channels, stride, bottleneck_ratio):
    
    if bottleneck_ratio is None or bottleneck_ratio == 1:
        block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                              nn.BatchNorm2d(out_channels))
    else:
        block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(out_channels, out_channels * bottleneck_ratio, kernel_size=1, bias=False),
                              nn.BatchNorm2d(out_channels * bottleneck_ratio))  
    return block


def shortcut_function(in_channels, out_channels, stride, bottleneck_ratio):
    
    if bottleneck_ratio is None:
        shortcut = lambda x: 0
    
    elif stride != 1 or in_channels != out_channels * bottleneck_ratio:
        shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * bottleneck_ratio, kernel_size=1, stride=stride, bias=False),
                                 nn.BatchNorm2d(out_channels * bottleneck_ratio))
    else:
        shortcut = lambda x: x
    
    return shortcut


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, bottleneck_ratio):

        super(ResidualBlock, self).__init__()

        self.F = residual_function(in_channels, out_channels, stride, bottleneck_ratio)
        self.shortcut = shortcut_function(in_channels, out_channels, stride, bottleneck_ratio)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        H = self.F(x) + self.shortcut(x)
        return self.relu(H)
    

class ResNet(nn.Module):

    def __init__(self, layer_config, bottleneck_ratio, num_classes=1000, first_in_channels=64):

        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, first_in_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(first_in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers = []
        in_channels = first_in_channels

        for out_channels, stride, n in layer_config: 
            for i in range (n):
                layers.append(ResidualBlock(in_channels, out_channels, stride if i==0 else 1, bottleneck_ratio))
                if bottleneck_ratio is None:
                    in_channels = out_channels
                else:
                    in_channels = out_channels * bottleneck_ratio

        self.convs = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.convs(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
