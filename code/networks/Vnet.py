import torch
from torch import nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    return nn.ELU(inplace=True) if elu else nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(nchan)
        self.relu1 = ELUCons(elu, nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


# Part1. return relu1(bn(conv(x))+(x,...,x))
class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x)) # torch.Size([2, 16, 512, 512])
        # 3 channel -> 1 channel
        if list(x.size())[1] == 3:
            x = x.sum(axis=1) / 3 # torch.Size([2, 512, 512])
            x = x.unsqueeze(1) # torch.Size([2, 1, 512, 512])
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16)) # torch.Size([2, 16, 512, 512])
        return out


# Part2. 
# down = relu1(bn(conv(x)))
# out = relu1(bn(conv(dropoutOrNot(down))))*nConvs
# return relu2(down + out)
class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.do1 = nn.Dropout2d() if dropout else passthrough
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


# Part3.
# xcat = cat(relu1(bn(upconv(doOrNot(x)))), do(skipx))
# out = relu1(bn(conv(dropoutOrNot(xcat))))*nConvs
# return relu2(xcat + out)
class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.do1 = nn.Dropout2d() if dropout else passthrough
        self.do2 = nn.Dropout2d()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(outChans // 2)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


# Part4.
# conv = conv2(relu1(bn1(conv1(x))))
# changechannelposition_flatten = conv.permute(0, 2, 3, 1).view(allnumber//2, 2)
# return softmax(changechannelposition_flatten)
class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.conv2 = nn.Conv2d(outChans, outChans, kernel_size=1)
        self.softmax = F.log_softmax if nll else F.softmax

        self.outChans = outChans

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        '''
        # make channels the last axis
        out = out.permute(0, 2, 3, 1).contiguous()
        # flatten
        out = out.view(out.numel() // self.outChans, self.outChans)
        # treat channel 0 as the predicted output
        out = self.softmax(out, dim=0)
        '''
        return out


class Vnet(nn.Module):
    def __init__(self, inChans, outChans, elu=True, nll=False):
        super(Vnet, self).__init__()
        self.in_tr      = InputTransition(inChans, 16, elu)
        self.down_tr32  = DownTransition(16, 1, elu)
        self.down_tr64  = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 2, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256   = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128   = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64    = UpTransition(128, 64, 1, elu)
        self.up_tr32    = UpTransition(64, 32, 1, elu)
        self.out_tr     = OutputTransition(32, outChans, elu, nll)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        en16  = self.in_tr(x)
        en32  = self.down_tr32(en16)
        en64  = self.down_tr64(en32)
        en128 = self.down_tr128(en64)
        en256 = self.down_tr256(en128)
        de256 = self.up_tr256(en256, en128)
        de128 = self.up_tr128(de256, en64)
        de64  = self.up_tr64(de128, en32)
        de32  = self.up_tr32(de64, en16)
        out   = self.out_tr(de32)
        return out
        # return self.activation(out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    bz, ch, och, image_size = 4, 1, 2, [512, 512]
    # batch_size, channel, w, h
    x = torch.Tensor(bz, ch, image_size[0], image_size[1]).to(device)
    print("x size: {}".format(x.size()))

    model = Vnet(ch, och).to(device)
    print(model)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))

    out = model(x)
    # out = out.view(bz, image_size[0], image_size[1], och).permute(0, 3, 1, 2)
    print("out size: {}".format(out.size()))
    print(out.sigmoid())
    print(nn.Softmax(dim=1)(out))