import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return F.relu(x)

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity
        return F.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_landmarks=5):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_landmarks * 2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """ ResNet의 블록을 생성하는 함수 """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def ResNet18(num_landmarks=5):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_landmarks)

def ResNet34(num_landmarks=5):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_landmarks)

def ResNet50(num_landmarks=5):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_landmarks)

def ResNet101(num_landmarks=5):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_landmarks)

class HourglassBlock(nn.Module):
    def __init__(self, in_channels, depth=4):
        super().__init__()
        self.depth = depth
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        for i in range(depth):
            self.down_layers.append(nn.Conv2d(in_channels * (2**i), in_channels * (2**(i+1)), kernel_size=3, stride=2, padding=1))
        
        for i in range(depth-1, -1, -1):
            self.up_layers.append(nn.ConvTranspose2d(in_channels * (2**(i+1)), in_channels * (2**i), kernel_size=3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        skip_connections = []
        for layer in self.down_layers:
            x = F.relu(layer(x))
            skip_connections.append(x)

        for i, layer in enumerate(self.up_layers):
            x = F.relu(layer(x))
            x += skip_connections[-(i+1)]

        return x

class DeepHourglassNet(nn.Module):
    def __init__(self, num_landmarks=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.hg1 = HourglassBlock(64, depth=5)
        self.hg2 = HourglassBlock(64, depth=5)
        self.hg3 = HourglassBlock(64, depth=5)
        self.fc = nn.Linear(64 * 96 * 96, num_landmarks * 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.hg1(x)
        x = self.hg2(x)
        x = self.hg3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DeepViT(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_landmarks=5, dim=768, depth=12, heads=12, mlp_dim=2048):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, batch_first=True),
            num_layers=depth
        )
        self.fc = nn.Linear(dim, num_landmarks * 2)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x += self.pos_embedding
        x = self.transformer(x)
        x = self.fc(x[:, 0])
        return x