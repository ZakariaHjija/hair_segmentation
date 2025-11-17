import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn


# for pool_size  = ( 1,1 ) for instance we obtain a tensor of size ( in_channels , 1 , 1 ) basically a 1-bin descriptor but since we will concatenate the tensor , along the channel dim ,  with other global priors of different pool_sizes ( 2,2 ).... we need to divide the number of channels of theses pooling layer by the number of diiferent scales in this case by 4

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
                          nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                          nn.BatchNorm2d(in_channels // len(pool_sizes)),
                          nn.ReLU(inplace=True))
            for pool_size in pool_sizes   
        ])



    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pooled_features = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        return torch.cat([x] + pooled_features, dim=1)


class PSPNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # ResNet152
        resnet = torchvision.models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        self.psp_module = PSPModule(in_channels=2048, pool_sizes=[1, 2, 3, 6])

        # Finale layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(2048 + 2048 , 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1) # necessary : gives 
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.psp_module(x)
        x = self.final_conv(x)

        # Upsample to original size
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=True)
        return torch.sigmoid(x)


