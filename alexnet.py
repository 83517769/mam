import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torch
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
#        self.classifier = nn.Sequential(
#            nn.Dropout(),
#            nn.Linear(256 * 6 * 6, 4096),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            nn.Linear(4096, 4096),
#            nn.ReLU(inplace=True),
#            nn.Linear(4096, num_classes),
#        )
        self.mil = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        )
        self.pooling = nn.MaxPool2d(6,stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.mil(x)
        feature_map = torch.sigmoid(x)
        x = self.pooling(feature_map)
        x = x.view(x.size(0),1)
#        x = x.view(x.size(0), 256 * 6 * 6)
#        x = self.classifier(x)
        return x, feature_map


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['alexnet'],model_root)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
#        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']), strict=False)
    return model
#        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
#    return model
