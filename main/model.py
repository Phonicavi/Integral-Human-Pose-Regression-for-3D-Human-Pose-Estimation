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
        x = self.deconv_layers(x)
        x = self.final_layer(x)

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

class ResidualBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class RegressModel(nn.Module):
    def __init__(self, joint_num,
                 num_stage=2,
                 p_dropout=0.5):
        self.inplanes = 2048
        self.midplanes = 1024
        self.outplanes = joint_num * cfg.depth_dim
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        super(RegressModel, self).__init__()

        # process input to linear size
        self.deconv_layers = self._make_deconv_layer(3)

        self.w1 = nn.Linear(self.midplanes, self.midplanes)
        self.batch_norm1 = nn.BatchNorm1d(self.midplanes)

        self.linear_stages = []
        for i in range(num_stage):
            self.linear_stages.append(ResidualBlock(self.midplanes, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.midplanes, self.outplanes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.midplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.midplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.midplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        # pre-processing
        print('step-0a', x.shape)
        y = self.deconv_layers(x)
        print('step-0b', y.shape)
        y = self.w1(y)
        print('step-0c', y.shape)
        y = self.batch_norm1(y)
        print('step-0d', y.shape)
        y = self.relu(y)
        print('step-0e', y.shape)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        print('step-0f', y.shape)
        y = self.w2(y)
        print('step-0g', y.shape)
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # print('step-01', x.shape)
        x = self.backbone(x)
        # print('step-02', x.shape)
        x = self.head(x)
        # print('step-03', x.shape)
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
    regress_net = RegressModel(joint_num)
    if is_train:
        backbone.init_weights()
        regress_net.init_weights()

    model = ResPoseNet(backbone, regress_net)
    return model
