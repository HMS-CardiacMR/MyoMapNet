import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size=[64, 64]):
        super(UNet, self).__init__()

        self.T1fitNet = nn.Sequential(
            nn.Linear(in_features=n_channels, out_features=400),
            nn.LeakyReLU(),
            nn.Linear(in_features=400, out_features=400),
            nn.LeakyReLU(),
            nn.Linear(in_features=400, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=1),

        )

        #below NN layers are not used in the MyoMapNet
        self.one_fcl = nn.Linear(100, 1)
        in_ch = 100
        inter_ch = 64

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, inter_ch, 5, padding=2),
            # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, inter_ch, 5, padding=2),
            # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 3, padding=1)
            # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
        )

    def forward(self, x, apply_conv=False):
        x = self.T1fitNet(x)
        return x


