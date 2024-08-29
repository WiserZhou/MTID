import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth=29, num_filters=None, output_dim=18, input_dim=1536, sequence_length=2):
        super(ResNet, self).__init__()

        # 设置默认的num_filters
        if num_filters is None:
            num_filters = [64, 128, 256, 512]

        # 确认深度符合Bottleneck
        assert (depth - 2) % 9 == 0, '当使用Bottleneck时，深度应为9n+2, 例如: 29, 47, 56, 110'
        n = (depth - 2) // 9

        # 嵌入层将输入映射到ResNet所需的形状
        self.embedding = nn.Linear(input_dim * sequence_length, num_filters[0] * 4 * 4)

        # 设置初始的inplanes值
        self.inplanes = num_filters[0]

        # 创建ResNet层
        self.layer1 = self._make_layer(Bottleneck, num_filters[1], n)
        self.layer2 = self._make_layer(Bottleneck, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(Bottleneck, num_filters[3], n, stride=2)

        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层输出
        self.fc = nn.Linear(num_filters[3] * Bottleneck.expansion, output_dim)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入x形状：[batch_size, sequence_length, feature_dim]
        batch_size, seq_len, feature_dim = x.size()
        

        # 将序列和特征维度展平为一个维度
        x = x.view(batch_size, -1)
        print(x.shape) # torch.Size([256, 3072])

        # 投影到ResNet输入需要的形状
        x = self.embedding(x)
        print(x.shape) # torch.Size([256, 1024])

        # 计算重塑尺寸
        reshape_size = int((x.size(1) // self.inplanes) ** 0.5)
        
        reshape_size = 32
        
        # 32*32 = 64+ 
        
        print(reshape_size)

        # 重塑为[batch_size, inplanes, calculated_height, calculated_width]
        x = x.view(batch_size, self.inplanes, reshape_size, reshape_size)
        print(x.shape)

        # 应用ResNet层
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)

        # 全局平均池化
        x = self.avgpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)  # 展平为[batch_size, num_filters[-1] * block.expansion]
        print(x.shape)

        # 应用最终的全连接层
        x = self.fc(x)
        print(x.shape)

        return x  # 输出形状: [batch_size, output_dim]
