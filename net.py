import torchvision.models as tvmodel
import torch.nn as nn
import torch


class YOLOv1_VGG(nn.Module):
    def __init__(self):
        super(YOLOv1_VGG, self).__init__()
        VGG = tvmodel.vgg16(pretrained=True)  # 调用预训练好的VGG16
        self.VGG = nn.Sequential(*list(VGG.children())[:-1])  #去掉全连接层

        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 7*7*30),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.VGG(input)
        input = input.view(input.size()[0], -1)
        input = self.Conn_layers(input)
        return input.reshape(-1, 30, 7, 7)

if __name__ == '__main__':
    x = torch.randn((1, 3, 448, 448))
    net = YOLOv1_VGG()
    print(net)
    y = net(x)
    print(y.size())