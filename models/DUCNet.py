
import torch
import torch.nn as nn
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):


    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = conv1x1(inplanes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print('identity', identity.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias = False)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class ResDUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):

        super(ResDUC, self).__init__()
        self.relu = nn.ReLU()
        downsample = nn.Sequential(
            conv1x1(inplanes, planes),
            nn.BatchNorm2d(planes),
        )
        self.conv = Bottleneck(inplanes, planes, downsample=downsample)
        # self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias = False)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        x = self.pixel_shuffle(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # print(x.shape)
        return x

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, conv_list):
        super(ASPP, self).__init__()
        self.conv_list = conv_list
        self.conv = nn.ModuleList([nn.Conv2d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias = False) for dil in conv_list])
        self.bn = nn.ModuleList([nn.BatchNorm2d(planes) for dil in conv_list])
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.bn[0](self.conv[0](x))
        for i in range(1, len(self.conv_list)):
            y += self.bn[i](self.conv[i](x))
        x = self.relu(y)

        return x

class DUCNet(nn.Module):

    def __init__(self, model):
        super(DUCNet, self).__init__()


        self.conv1 = model.conv1
        self.bn0 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        # self.duc5 = DUC(64, 64*2)

        self.ASPP = ASPP(64, 64, [1, 3, 5, 7])
        # self.ASPP = ASPP(32, 64, [1, 3, 5, 7])
        self.ASPPout = nn.Conv2d(64, 1, 1)

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)
        # nn.init.kaiming_normal_(self.transformer, mode='fan_out', nonlinearity='relu')



    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        dfm1 = fm3 + self.duc1(fm4)

        dfm2 = fm2 + self.duc2(dfm1)

        dfm3 = fm1 + self.duc3(dfm2)

        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))

        dfm4 = conv_x + self.duc4(dfm3_t)
        # print('dfm4', dfm4.shape)
        # dfm5 = self.duc5(dfm4)
        # print('dfm5', dfm5.shape)
        out = self.ASPP(dfm4)
        # print('ASPP', out.shape)
        out = self.ASPPout(out)
        # print('ASPPout', out.shape)
        out = self.relu(out)
        return out

class ResDUCNet(nn.Module):

    def __init__(self, model):
        super(ResDUCNet, self).__init__()


        self.conv1 = model.conv1
        self.bn0 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.duc1 = ResDUC(512, 1024)
        self.duc2 = ResDUC(256, 512)
        self.duc3 = ResDUC(128, 256)
        self.duc4 = ResDUC(32, 64)
        # self.duc5 = DUC(64, 64*2)

        self.ASPP = ASPP(64, 64, [1, 3, 5, 7])
        # self.ASPP = ASPP(32, 64, [1, 3, 5, 7])
        self.ASPPout = nn.Conv2d(64, 1, 1)

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)

        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        # print('fm3', fm3.shape)

        dfm1 = fm3 + self.duc1(fm4)

        dfm2 = fm2 + self.duc2(dfm1)

        dfm3 = fm1 + self.duc3(dfm2)

        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        # print('dfm3_t', dfm3_t.shape)

        dfm4 = conv_x + self.duc4(dfm3_t)
        # print('dfm4', dfm4.shape)
        # dfm5 = self.duc5(dfm4)
        # print('dfm5', dfm5.shape)
        out = self.ASPP(dfm4)
        # print('ASPP', out.shape)
        out = self.ASPPout(out)
        # print('ASPPout', out.shape)
        out = self.relu(out)

        # out = self.bilinear(out)
        return out


class MADUCNet(nn.Module):

    def __init__(self, model):
        super(MADUCNet, self).__init__()


        self.conv1 = model.conv1
        self.bn0 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        # self.duc5 = DUC(64, 64*2)

        self.ASPP1 = ASPP(1024, 1024, [1, 3, 5, 7])
        self.ASPP2 = ASPP(512, 512, [1, 3, 5, 7])
        self.ASPP3 = ASPP(128, 128, [1, 3, 5, 7])
        self.ASPP4 = ASPP(64, 64, [1, 3, 5, 7])
        self.ASPPout = nn.Conv2d(64, 1, 1)

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        dfm1 = fm3 + self.duc1(fm4)
        # print('dfm1', dfm1.shape)
        dfm1 = self.ASPP1(dfm1)

        dfm2 = fm2 + self.duc2(dfm1)
        dfm2 = self.ASPP2(dfm2)

        dfm3 = fm1 + self.duc3(dfm2)


        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        dfm3_t = self.ASPP3(dfm3_t)

        dfm4 = conv_x + self.duc4(dfm3_t)
        # print('dfm4', dfm4.shape)
        # dfm5 = self.duc5(dfm4)
        # print('dfm5', dfm5.shape)
        out = self.ASPP4(dfm4)
        # print('ASPP', out.shape)
        out = self.ASPPout(out)
        # print('ASPPout', out.shape)
        out = self.relu(out)
        return out

if __name__ == '__main__':
    model = ResDUCNet(model=models.resnet50(pretrained=False))
    out = model(torch.rand(1,3,224,288))
    print(out.shape)
