import torch.nn as nn
from nets.resnet import ResNetBackbone
from config import cfg

class HeadNet(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 256

        super(HeadNet, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        #print('[Head][Before deconv]', x.shape)
        x = self.deconv_layers(x)
        #print('[Head][After deconv]', x.shape)
        x = self.final_layer(x)
        #print('[Head][After final]', x.shape)

        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class RegressNet(nn.Module):

    def __init__(self, joint_num, p_dropout=0.5):
        self.inplanes = 2048
        self.midplanes = 1024
        self.outplanes = joint_num * cfg.depth_dim
        self.p_dropout = p_dropout

        super(RegressNet, self).__init__()

        self.pooling = nn.AvgPool2d(8)

        self.w1 = nn.Linear(self.inplanes, self.midplanes)
        self.batch_norm1 = nn.BatchNorm1d(self.midplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.w2 = nn.Linear(self.midplanes, self.outplanes)

    def forward(self, x):

        print('[1a-before pooling]', x.shape)
        x = self.pooling(x)
        print('[1b-after pooling]', x.shape)
        x = self.w1(x)
        print('[2a-after w1]', x.shape)
        x = self.batch_norm1(x)
        print('[2b-after bn1]', x.shape)
        x = self.relu(x)
        print('[2c-after relu]', x.shape)
        x = self.dropout(x)
        print('[3a-before w2]', x.shape)
        x = self.w2(x)
        print('[3b-after w2]', x.shape)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        #print('[Before backbone]', x.shape)
        x = self.backbone(x)
        #print('[After backbone]', x.shape)
        x = self.head(x)
        #print('[After head]', x.shape)
        return x

def get_pose_net(cfg, is_train, joint_num):
    
    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = HeadNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()

    model = ResPoseNet(backbone, head_net)
    return model

def get_pose_net_baseline(cfg, is_train, joint_num):

    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = RegressNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()

    model = ResPoseNet(backbone, head_net)
    return model