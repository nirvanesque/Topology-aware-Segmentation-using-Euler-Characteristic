import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class UNet2D(nn.Module):

    def __init__(self, num_classes, num_channel=1):
        super(UNet2D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(num_channel, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv32 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv22 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv13 = nn.Conv2d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv2d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))

        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        x = self.conv14(x1)
        return x


class UNet3D(nn.Module):

    def __init__(self, num_classes, num_channel=1):
        super(UNet3D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(num_channel, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv3d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        # odd e.g. in_size 11 -> out_size = upper(11/2) -> 6
        # even e.g., in_size=12 -> out_size = 12/2 -> 6

        self.conv21 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)

        self.conv31 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)

        self.conv41 = nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=use_bias)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = nn.Conv3d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv3d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)


    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        self.neck = x4
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        x = self.conv14(x1)

        return x

    def get_neck(self):
        return self.neck


    # x_filter = np.array([[1, 0, -1],
    #                      [2, 0, -2],
    #                      [1, 0, -1]]).reshape(1, 1, 3, 3)
    #
    # x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
    # x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
    # conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
    #                    dilation=1, bias=False)
    #
    # conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())
    #
    # y_filter = np.array([[1, 2, 1],
    #                      [0, 0, 0],
    #                      [-1, -2, -1]]).reshape(1, 1, 3, 3)
    # y_filter = np.repeat(y_filter, axis=1, repeats=object_classes)
    # y_filter = np.repeat(y_filter, axis=0, repeats=object_classes)
    # conv_y = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
    #                    bias=False)
    # conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())
    #
    # conv_x = conv_x.to(input.device)
    # conv_y = conv_y.to(input.device)
    # for param in conv_y.parameters():
    #     param.requires_grad = False
    # for param in conv_x.parameters():
    #     param.requires_grad = False


class EC(nn.Module):
    def __init__(self, subpatch_size=32, subpatch_stride=8):
        super(EC, self).__init__()

        self.subpatch_size = subpatch_size
        self.overlap_size = subpatch_stride
        self.isApproxBinary = True

        self.CeConv_1 = nn.Conv2d(1, 4, kernel_size=2, bias=False)
        self.CeConv_2 = nn.Conv2d(1, 4, kernel_size=2, bias=False)
        self.CeConv_3 = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        # [4, 2, 2], [4, 2, 2], [2, 2, 2]
        x_filter_1 = np.array([[[1, -1], [-1, -1]],
                             [[-1, 1], [-1, -1]],
                             [[-1, -1], [-1, 1]],
                             [[-1, -1], [1, -1]],]
                             ).reshape(4, 1, 2, 2)

        x_filter_2 = np.array([[[1, 1], [1, -1]],
                               [[1, 1], [-1, 1]],
                               [[-1, 1], [1, 1]],
                               [[1, -1], [1, 1]]
                               ]).reshape(4, 1, 2, 2)

        x_filter_3 = np.array([[[1, -1], [-1, 1]],
                               [[-1, 1], [1, -1]],
                               ]).reshape(2, 1, 2, 2)


        self.CeConv_1.weight = nn.Parameter(torch.from_numpy(x_filter_1).float())
        self.CeConv_2.weight = nn.Parameter(torch.from_numpy(x_filter_2).float())
        self.CeConv_3.weight = nn.Parameter(torch.from_numpy(x_filter_3).float())

        for param in self.CeConv_1.parameters():
            param.requires_grad = False
        for param in self.CeConv_2.parameters():
            param.requires_grad = False
        for param in self.CeConv_3.parameters():
            param.requires_grad = False#False

        self.AvePool = nn.AvgPool2d(self.subpatch_size, stride=self.overlap_size)


    def forward(self, x_hard, pad_value=-1, isReturnEcMap=True, isReturnInternalFeatures=False):
        return_dist = {}
        p2d = (1,1,1,1)

        '''EC hard'''
        x_norm_hard = x_hard * 2 -1

        x_norm_pad_hard = nn.functional.pad(x_norm_hard, p2d, 'constant', pad_value)
        '''1.1 EC_map unnormalized'''
        f1_hard_norm = self.CeConv_1(x_norm_pad_hard)
        f2_hard_norm = self.CeConv_2(x_norm_pad_hard)       # if x is binary, feature 2 should either be 0, 1/3, 2/3 or 1, and only count the number of 1
        f3_hard_norm = self.CeConv_3(x_norm_pad_hard)
        '''1.2 EC_map_normalized'''
        f1_hard_norm[f1_hard_norm <= 3.9] = 0
        f2_hard_norm[f2_hard_norm <= 3.9] = 0
        f3_hard_norm[f3_hard_norm <= 3.9] = 0
        f1_hard_norm = f1_hard_norm/4
        f2_hard_norm = f2_hard_norm / 4
        f3_hard_norm = f3_hard_norm / 4

        '''1.3 EC_NUMBER'''
        EC_hard_1 = f1_hard_norm.sum(dim=(1, 2, 3))
        EC_hard_2 = f2_hard_norm.sum(dim=(1, 2, 3))
        EC_hard_3 = f3_hard_norm.sum(dim=(1, 2, 3))
        EC_hard = 1/4 * (EC_hard_1 - EC_hard_2 - 2 * EC_hard_3)

        return_dist.update({'EC': EC_hard})

        if isReturnInternalFeatures:
            return_dist.update({'EC_features': [f1_hard_norm.sum(dim=1), f2_hard_norm.sum(dim=1), f3_hard_norm.sum(dim=1)]})

        '''1.4 EC numbers on subpatch'''
        if isReturnEcMap:
            EC_hard_patch_1 = self.subpatch_size * self.subpatch_size * self.AvePool(f1_hard_norm.sum(dim=1))
            EC_hard_patch_2 = self.subpatch_size * self.subpatch_size * self.AvePool(f2_hard_norm.sum(dim=1))
            EC_hard_patch_3 = self.subpatch_size * self.subpatch_size * self.AvePool(f3_hard_norm.sum(dim=1))

            EC_hard_patch = 1/4 * (EC_hard_patch_1 - EC_hard_patch_2 - 2 * EC_hard_patch_3)
            return_dist.update({'EC_map': EC_hard_patch})

        return return_dist


class EC_3d(nn.Module):
    def __init__(self, subpatch_size=32, subpatch_stride=8, ):

        super(EC_3d, self).__init__()
        # self.device = device
        # self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        # self.input_dim = 1
        # self.output_dim_1 = 4
        # self.output_dim_2 = 2

        self.subpatch_size = subpatch_size#16
        self.overlap_size = subpatch_stride
        self.thred = 0.9


        # self.CeConv_1 = nn.Conv3d(1, 4, kernel_size=2, bias=False)
        # self.CeConv_2 = nn.Conv3d(1, 4, kernel_size=2, bias=False)
        # self.CeConv_3 = nn.Conv3d(1, 2, kernel_size=2, bias=False)
        # [4, 2, 2], [4, 2, 2], [2, 2, 2]
        # x_filter_1 = np.array([[[1, -1], [-1, -1]],
        #                      [[-1, 1], [-1, -1]],
        #                      [[-1, -1], [-1, 1]],
        #                      [[-1, -1], [1, -1]],]
        #                      ).reshape(4, 1, 2, 2)
        #
        # x_filter_2 = np.array([[[1, 1], [1, -1]],
        #                        [[1, 1], [-1, 1]],
        #                        [[-1, 1], [1, 1]],
        #                        [[1, -1], [1, 1]]
        #                        ]).reshape(4, 1, 2, 2)
        #
        # x_filter_3 = np.array([[[1, -1], [-1, 1]],
        #                        [[-1, 1], [1, -1]],
        #                        ]).reshape(2, 1, 2, 2)

        conv_channels = [8,
                         12, 12, 4,
                         24, 24, 8,
                         6, 8, 24, 24, 6, 2,
                         24, 24, 8,
                         12, 12, 4,
                         8, ]

        weights = [0.12500,
                   0.00000, -0.25000, -0.75000,
                   -0.12500, -0.37500, -0.12500,
                   0.00000, -0.25000, -0.25000, 0.00000, 0.00000, 0.00000,
                   -0.12500, 0.12500, 0.37500,
                   0.00000, 0.25000, 0.25000,
                   0.12500]

        ''' 1.1 Define kernel features. Filter index is as shown in slides'''
        if True:
            x_filters = []
            '''filter 1: one positive value'''
            x_filters.append(np.array([
                [[[1, 0], [0, 0]], [[0, 0], [0, 0]],],
                [[[0, 1], [0, 0]], [[0, 0], [0, 0]],],
                [[[0, 0], [1, 0]], [[0, 0], [0, 0]], ],
                [[[0, 0], [0, 1]], [[0, 0], [0, 0]], ],
                [[[0, 0], [0, 0]], [[1, 0], [0, 0]], ],
                [[[0, 0], [0, 0]], [[0, 1], [0, 0]], ],
                [[[0, 0], [0, 0]], [[0, 0], [1, 0]], ],
                [[[0, 0], [0, 0]], [[0, 0], [0, 1]], ],]).reshape(8, 1, 2, 2, 2))

            '''filter 2.1: two positive values'''
            x_filters.append(np.array([
                    [[[1, 1], [0, 0]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[0, 0], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[0, 0], [0, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 0], [0, 0]], [[0, 1], [0, 1]], ],

                    [[[1, 0], [0, 0]], [[1, 0], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [0, 1]], [[0, 0], [0, 1]], ],
                    ]).reshape(12, 1, 2, 2, 2))

            '''filters 2.2: two positive values'''
            x_filters.append(np.array([
                    [[[1, 0], [0, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 1], [1, 0]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 0], [0, 0]], [[0, 1], [1, 0]], ],

                    [[[1, 0], [0, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[1, 0], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 1], [0, 0]], ],

                    [[[1, 0], [0, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[1, 0], [0, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 0], [1, 0]], ],
                ]).reshape(12, 1, 2, 2, 2))

            '''filters 2.3: two positive values'''
            x_filters.append(np.array([
                    [[[1, 0], [0, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 1], [0, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 0], [0, 1]], [[1, 0], [0, 0]], ],

                ]).reshape(4, 1, 2, 2, 2))

            '''filters 3.1: three positive values'''
            x_filters.append(np.array([
                    [[[1, 1], [1, 0]], [[0, 0], [0, 0]], ],
                    [[[1, 1], [0, 1]], [[0, 0], [0, 0]], ],
                    [[[1, 0], [1, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 1], [1, 1]], [[0, 0], [0, 0]], ],

                    [[[0, 0], [0, 0]], [[1, 1], [1, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 1], [0, 1]], ],
                    [[[0, 0], [0, 0]], [[1, 0], [1, 1]], ],
                    [[[0, 0], [0, 0]], [[0, 1], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[1, 0], [0, 0]], ],
                    [[[1, 0], [1, 0]], [[0, 0], [1, 0]], ],
                    [[[1, 0], [0, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[1, 0], [1, 0]], ],

                    [[[0, 1], [0, 1]], [[0, 1], [0, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 0], [0, 1]], ],
                    [[[0, 1], [0, 0]], [[0, 1], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 1], [0, 1]], ],

                    [[[1, 1], [0, 0]], [[1, 0], [0, 0]], ],
                    [[[1, 1], [0, 0]], [[0, 1], [0, 0]], ],
                    [[[1, 0], [0, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[1, 1], [0, 0]], ],

                    [[[0, 0], [1, 1]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [1, 0]], [[0, 0], [1, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 0], [1, 1]], ],
                ]).reshape(24, 1, 2, 2, 2))

            '''filters 3.2: three positive values'''
            x_filters.append(np.array([
                    [[[1, 1], [0, 0]], [[0, 0], [0, 1]], ],
                    [[[1, 1], [0, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 1]], [[1, 0], [0, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 1], [0, 0]], ],

                    [[[0, 0], [1, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 0], [0, 1]], [[1, 1], [0, 0]], ],
                    [[[1, 0], [0, 0]], [[0, 0], [1, 1]], ],
                    [[[0, 1], [0, 0]], [[0, 0], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[0, 1], [0, 0]], ],
                    [[[1, 0], [1, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 1], [0, 1]], [[1, 0], [0, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 0], [1, 0]], ],

                    [[[0, 1], [0, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 0], [0, 1]], [[1, 0], [1, 0]], ],
                    [[[1, 0], [0, 0]], [[0, 1], [0, 1]], ],
                    [[[0, 0], [1, 0]], [[0, 1], [0, 1]], ],

                    [[[1, 0], [0, 1]], [[1, 0], [0, 0]], ],
                    [[[1, 0], [0, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 1], [1, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[0, 1], [1, 0]], ],

                    [[[0, 1], [1, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 1], [1, 0]], ],
                    [[[1, 0], [0, 1]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[1, 0], [0, 1]], ],
                ]).reshape(24, 1, 2, 2, 2))

            '''filters 3.3: three positive values'''
            x_filters.append(np.array([
                    [[[0, 1], [1, 0]], [[1, 0], [0, 0]], ],
                    [[[1, 0], [0, 1]], [[0, 1], [0, 0]], ],
                    [[[1, 0], [0, 1]], [[0, 0], [1, 0]], ],
                    [[[0, 1], [1, 0]], [[0, 0], [0, 1]], ],

                    [[[1, 0], [0, 0]], [[0, 1], [1, 0]], ],
                    [[[0, 1], [0, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 0], [1, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 1], [1, 0]], ],

                ]).reshape(8, 1, 2, 2, 2))

            '''filters 4.1 four positive values'''
            x_filters.append(np.array([
                    [[[1, 1], [1, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 1], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 1], [0, 1]], ],

                    [[[1, 1], [0, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 0], [1, 1]], ],
                ]).reshape(6, 1, 2, 2, 2))

            '''filters 4.2 four positive values'''
            x_filters.append(np.array([
                    [[[1, 1], [1, 0]], [[1, 0], [0, 0]], ],     # 1, 2, 3, 5
                    [[[1, 1], [0, 1]], [[0, 1], [0, 0]], ],     # 1, 2, 4, 6
                    [[[1, 0], [1, 1]], [[0, 0], [1, 0]], ],     # 1, 3, 4, 7
                    [[[0, 1], [1, 1]], [[0, 0], [0, 1]], ],     # 2, 3, 4, 8

                    [[[1, 0], [0, 0]], [[1, 1], [1, 0]], ],     # 1, 5, 6, 7
                    [[[0, 1], [0, 0]], [[1, 1], [0, 1]], ],     # 2, 5, 6, 8
                    [[[0, 0], [1, 0]], [[1, 0], [1, 1]], ],     # 3, 5, 7, 8
                    [[[0, 0], [0, 1]], [[0, 1], [1, 1]], ],     # 4, 6, 7 ,8

                ]).reshape(8, 1, 2, 2, 2))

            '''filters 4.3 four positive values'''
            filter_4_3_double = np.array([

                    # '''1 2 3 4'''
                    [[[1, 1], [1, 0]], [[0, 1], [0, 0]], ],     # 1 2 3 x 6
                    [[[1, 1], [0, 1]], [[1, 0], [0, 0]], ],     # 1 2 x 4 5
                    [[[1, 0], [1, 1]], [[1, 0], [0, 0]], ],     # 1 x 3 4 5
                    [[[0, 1], [1, 1]], [[0, 1], [0, 0]], ],     # x 2 3 4 6

                    [[[1, 1], [1, 0]], [[0, 0], [1, 0]], ],  # 1 2 3 x 7
                    [[[1, 1], [0, 1]], [[0, 0], [0, 1]], ],  # 1 2 x 4 8
                    [[[1, 0], [1, 1]], [[0, 0], [0, 1]], ],  # 1 x 3 4 8
                    [[[0, 1], [1, 1]], [[0, 0], [1, 0]], ],  # x 2 3 4 7

                    # '''5 6 7 8'''
                    [[[0, 1], [0, 0]], [[1, 1], [1, 0]], ],     # 5 6 7 x 2
                    [[[1, 0], [0, 0]], [[1, 1], [0, 1]], ],     # 5 6 x 8 1
                    [[[1, 0], [0, 0]], [[1, 0], [1, 1]], ],     # 5 x 7 8 1
                    [[[0, 1], [0, 0]], [[0, 1], [1, 1]], ],     # x 6 7 8 2

                    [[[0, 0], [1, 0]], [[1, 1], [1, 0]], ],  # 5 6 7 x 3
                    [[[0, 0], [0, 1]], [[1, 1], [0, 1]], ],  # 5 6 x 8 4
                    [[[0, 0], [0, 1]], [[1, 0], [1, 1]], ],  # 5 x 7 8 4
                    [[[0, 0], [1, 0]], [[0, 1], [1, 1]], ],  # x 6 7 8 3


                    # '''1 3 5 7'''
                    [[[1, 0], [1, 1]], [[1, 0], [0, 0]], ],     # 1 3 5 x 4
                    [[[1, 1], [1, 0]], [[0, 0], [1, 0]], ],     # 1 3 x 7 2
                    [[[1, 1], [0, 0]], [[1, 0], [1, 0]], ],     # 1 x 5 7 2
                    [[[0, 0], [1, 1]], [[1, 0], [1, 0]], ],     # x 3 5 7 4

                    [[[1, 0], [1, 0]], [[1, 1], [0, 0]], ],  # 1 3 5 x 6
                    [[[1, 0], [1, 0]], [[0, 0], [1, 1]], ],  # 1 3 x 7 8
                    [[[1, 0], [0, 0]], [[1, 0], [1, 1]], ],  # 1 x 5 7 8
                    [[[0, 0], [1, 0]], [[1, 1], [1, 0]], ],  # x 3 5 7 6

                    # '''2 4 6 8'''
                    [[[0, 1], [1, 1]], [[0, 1], [0, 0]], ],     #  2 4 6 x 3
                    [[[1, 1], [0, 1]], [[0, 0], [0, 1]], ],     #  2 4 x 8 1
                    [[[1, 1], [0, 0]], [[0, 1], [0, 1]], ],     #  2 x 6 8 1
                    [[[0, 0], [1, 1]], [[0, 1], [0, 1]], ],     #  x 4 6 8 3

                    [[[0, 1], [0, 1]], [[1, 1], [0, 0]], ],  # 2 4 6 x 5
                    [[[0, 1], [0, 1]], [[0, 0], [1, 1]], ],  # 2 4 x 8 7
                    [[[0, 1], [0, 0]], [[0, 1], [1, 1]], ],  # 2 x 6 8 7
                    [[[0, 0], [0, 1]], [[1, 1], [0, 1]], ],  # x 4 6 8 5

                    # '''1 2 5 6'''
                    [[[1, 1], [0, 1]], [[1, 0], [0, 0]], ],     # 1 2 5 x 4
                    [[[1, 1], [1, 0]], [[0, 1], [0, 0]], ],     # 1 2 x 6 3
                    [[[1, 0], [1, 0]], [[1, 1], [0, 0]], ],     # 1 x 5 6 3
                    [[[0, 1], [0, 1]], [[1, 1], [0, 0]], ],     # x 2 5 6 4

                    [[[1, 1], [0, 0]], [[1, 0], [1, 0]], ],  # 1 2 5 x 7
                    [[[1, 1], [0, 0]], [[0, 1], [0, 1]], ],  # 1 2 x 6 8
                    [[[1, 0], [0, 0]], [[1, 1], [0, 1]], ],  # 1 x 5 6 8
                    [[[0, 1], [0, 0]], [[1, 1], [1, 0]], ],  # x 2 5 6 7

                    # '''3 4 7 8'''
                    [[[0, 1], [1, 1]], [[0, 0], [1, 0]], ],     # 3 4 7 x 2
                    [[[1, 0], [1, 1]], [[0, 0], [0, 1]], ],     # 3 4 x 8 1
                    [[[1, 0], [1, 0]], [[0, 0], [1, 1]], ],     # 3 x 7 8 1
                    [[[0, 1], [0, 1]], [[0, 0], [1, 1]], ],     # x 4 7 8 2

                    [[[0, 0], [1, 1]], [[1, 0], [1, 0]], ],  # 3 4 7 x 5
                    [[[0, 0], [1, 1]], [[0, 1], [0, 1]], ],  # 3 4 x 8 6
                    [[[0, 0], [1, 0]], [[0, 1], [1, 1]], ],  # 3 x 7 8 6
                    [[[0, 0], [0, 1]], [[1, 0], [1, 1]], ],  # x 4 7 8 5
                ]).reshape(48, 1, 2, 2, 2)
            filter_4_3 = np.unique(filter_4_3_double, axis=0)
            assert filter_4_3.shape[0] == 24
            x_filters.append(filter_4_3)
            # this filter is doubled, need to unique to 24 features

            '''filters 4.4 four positive values'''
            x_filters.append(np.array([
                    # '''1 2 3 4'''
                    [[[1, 1], [1, 0]], [[0, 0], [0, 1]], ],     # 1 2 3 x 8
                    [[[1, 1], [0, 1]], [[0, 0], [1, 0]], ],     # 1 2 x 4 7
                    [[[1, 0], [1, 1]], [[0, 1], [0, 0]], ],     # 1 x 3 4 6
                    [[[0, 1], [1, 1]], [[1, 0], [0, 0]], ],     # x 2 3 4 5
                    # '''5 6 7 8'''
                    [[[0, 0], [0, 1]], [[1, 1], [1, 0]], ],     # 5 6 7 x 4
                    [[[0, 0], [1, 0]], [[1, 1], [0, 1]], ],     # 5 6 x 8 3
                    [[[0, 1], [0, 0]], [[1, 0], [1, 1]], ],     # 5 x 7 8 2
                    [[[1, 0], [0, 0]], [[0, 1], [1, 1]], ],     # x 6 7 8 1

                    # '''1 3 5 7'''
                    [[[1, 0], [1, 0]], [[1, 0], [0, 1]], ],     # 1 3 5 x 8
                    [[[1, 0], [1, 0]], [[0, 1], [1, 0]], ],     # 1 3 x 7 6
                    [[[1, 0], [0, 1]], [[1, 0], [1, 0]], ],     # 1 x 5 7 4
                    [[[0, 1], [1, 0]], [[1, 0], [1, 0]], ],     # x 3 5 7 2
                    # '''2 4 6 8'''
                    [[[0, 1], [0, 1]], [[0, 1], [1, 0]], ],     #  2 4 6 x 7
                    [[[0, 1], [0, 1]], [[1, 0], [0, 1]], ],     #  2 4 x 8 5
                    [[[0, 1], [1, 0]], [[0, 1], [0, 1]], ],     #  2 x 6 8 3
                    [[[1, 0], [0, 1]], [[0, 1], [0, 1]], ],     #  x 4 6 8 1

                    # '''1 2 5 6'''
                    [[[1, 1], [0, 0]], [[1, 0], [0, 1]], ],     # 1 2 5 x 8
                    [[[1, 1], [0, 0]], [[0, 1], [1, 0]], ],     # 1 2 x 6 7
                    [[[1, 0], [0, 1]], [[1, 1], [0, 0]], ],     # 1 x 5 6 4
                    [[[0, 1], [1, 0]], [[1, 1], [0, 0]], ],     # x 2 5 6 3
                    # '''3 4 7 8'''
                    [[[0, 0], [1, 1]], [[0, 1], [1, 0]], ],     # 3 4 7 x 6
                    [[[0, 0], [1, 1]], [[1, 0], [0, 1]], ],     # 3 4 x 8 5
                    [[[0, 1], [1, 0]], [[0, 0], [1, 1]], ],     # 3 x 7 8 2
                    [[[1, 0], [0, 1]], [[0, 0], [1, 1]], ],     # x 4 7 8 1
                ]).reshape(24, 1, 2, 2, 2))

            '''filters 4.5 four positive values'''
            x_filters.append(np.array([
                    [[[1, 1], [0, 0]], [[0, 0], [1, 1]], ],     # 12 78
                    [[[0, 0], [1, 1]], [[1, 1], [0, 0]], ],     # 34 56

                    [[[1, 0], [1, 0]], [[0, 1], [0, 1]], ],     # 13 68
                    [[[0, 1], [0, 1]], [[1, 0], [1, 0]], ],     # 24 57

                    [[[1, 0], [0, 1]], [[1, 0], [0, 1]], ],     # 15 48
                    [[[0, 1], [1, 0]], [[0, 1], [1, 0]], ],     # 26 37

                ]).reshape(6, 1, 2, 2, 2))

            '''filters 4.6 four positive values'''
            x_filters.append(np.array([
                    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], ],     # 1 4 6 7
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]], ],     # 2 3 5 8

                ]).reshape(2, 1, 2, 2, 2))

            '''filters 5.1: five positive values'''
            x_filters.append((1 - np.array([
                    [[[1, 1], [1, 0]], [[0, 0], [0, 0]], ],
                    [[[1, 1], [0, 1]], [[0, 0], [0, 0]], ],
                    [[[1, 0], [1, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 1], [1, 1]], [[0, 0], [0, 0]], ],

                    [[[0, 0], [0, 0]], [[1, 1], [1, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 1], [0, 1]], ],
                    [[[0, 0], [0, 0]], [[1, 0], [1, 1]], ],
                    [[[0, 0], [0, 0]], [[0, 1], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[1, 0], [0, 0]], ],
                    [[[1, 0], [1, 0]], [[0, 0], [1, 0]], ],
                    [[[1, 0], [0, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[1, 0], [1, 0]], ],

                    [[[0, 1], [0, 1]], [[0, 1], [0, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 0], [0, 1]], ],
                    [[[0, 1], [0, 0]], [[0, 1], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 1], [0, 1]], ],

                    [[[1, 1], [0, 0]], [[1, 0], [0, 0]], ],
                    [[[1, 1], [0, 0]], [[0, 1], [0, 0]], ],
                    [[[1, 0], [0, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[1, 1], [0, 0]], ],

                    [[[0, 0], [1, 1]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [1, 0]], [[0, 0], [1, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 0], [1, 1]], ],
                ])).reshape(24, 1, 2, 2, 2))

            '''filters 5.2: five positive values'''
            x_filters.append((1 - np.array([
                    [[[1, 1], [0, 0]], [[0, 0], [0, 1]], ],
                    [[[1, 1], [0, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 1]], [[1, 0], [0, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 1], [0, 0]], ],

                    [[[0, 0], [1, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 0], [0, 1]], [[1, 1], [0, 0]], ],
                    [[[1, 0], [0, 0]], [[0, 0], [1, 1]], ],
                    [[[0, 1], [0, 0]], [[0, 0], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[0, 1], [0, 0]], ],
                    [[[1, 0], [1, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 1], [0, 1]], [[1, 0], [0, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 0], [1, 0]], ],

                    [[[0, 1], [0, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 0], [0, 1]], [[1, 0], [1, 0]], ],
                    [[[1, 0], [0, 0]], [[0, 1], [0, 1]], ],
                    [[[0, 0], [1, 0]], [[0, 1], [0, 1]], ],

                    [[[1, 0], [0, 1]], [[1, 0], [0, 0]], ],
                    [[[1, 0], [0, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 1], [1, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[0, 1], [1, 0]], ],

                    [[[0, 1], [1, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 1], [1, 0]], ],
                    [[[1, 0], [0, 1]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[1, 0], [0, 1]], ],
                ])).reshape(24, 1, 2, 2, 2))

            '''filters 5.3: five positive values'''
            x_filters.append((1 - np.array([
                    [[[0, 1], [1, 0]], [[1, 0], [0, 0]], ],
                    [[[1, 0], [0, 1]], [[0, 1], [0, 0]], ],
                    [[[1, 0], [0, 1]], [[0, 0], [1, 0]], ],
                    [[[0, 1], [1, 0]], [[0, 0], [0, 1]], ],

                    [[[1, 0], [0, 0]], [[0, 1], [1, 0]], ],
                    [[[0, 1], [0, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 0], [1, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 1], [1, 0]], ],

                ])).reshape(8, 1, 2, 2, 2))

            '''filter 6.1: six positive values'''
            x_filters.append((1 - np.array([
                    [[[1, 1], [0, 0]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [1, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 1], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[0, 0], [1, 1]], ],

                    [[[1, 0], [1, 0]], [[0, 0], [0, 0]], ],
                    [[[0, 1], [0, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 0], [1, 0]], ],
                    [[[0, 0], [0, 0]], [[0, 1], [0, 1]], ],

                    [[[1, 0], [0, 0]], [[1, 0], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [0, 1]], [[0, 0], [0, 1]], ],
                    ])).reshape(12, 1, 2, 2, 2))

            '''filters 6.2: six positive values'''
            x_filters.append((1 - np.array([
                    [[[1, 0], [0, 1]], [[0, 0], [0, 0]], ],
                    [[[0, 1], [1, 0]], [[0, 0], [0, 0]], ],
                    [[[0, 0], [0, 0]], [[1, 0], [0, 1]], ],
                    [[[0, 0], [0, 0]], [[0, 1], [1, 0]], ],

                    [[[1, 0], [0, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[1, 0], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 1], [0, 0]], ],

                    [[[1, 0], [0, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 1], [0, 0]], [[1, 0], [0, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 0], [0, 1]], [[0, 0], [1, 0]], ],
                ])).reshape(12, 1, 2, 2, 2))

            '''filters 6.3: six positive values'''
            x_filters.append((1 - np.array([
                    [[[1, 0], [0, 0]], [[0, 0], [0, 1]], ],
                    [[[0, 1], [0, 0]], [[0, 0], [1, 0]], ],
                    [[[0, 0], [1, 0]], [[0, 1], [0, 0]], ],
                    [[[0, 0], [0, 1]], [[1, 0], [0, 0]], ],

                ])).reshape(4, 1, 2, 2, 2))

            '''filters 7: seven positive values'''
            x_filters.append((1 - np.array([
                [[[1, 0], [0, 0]], [[0, 0], [0, 0]],],
                [[[0, 1], [0, 0]], [[0, 0], [0, 0]],],
                [[[0, 0], [1, 0]], [[0, 0], [0, 0]], ],
                [[[0, 0], [0, 1]], [[0, 0], [0, 0]], ],
                [[[0, 0], [0, 0]], [[1, 0], [0, 0]], ],
                [[[0, 0], [0, 0]], [[0, 1], [0, 0]], ],
                [[[0, 0], [0, 0]], [[0, 0], [1, 0]], ],
                [[[0, 0], [0, 0]], [[0, 0], [0, 1]], ],])).reshape(8, 1, 2, 2, 2))

        ''' 1.2 Delete the kernels with 0 weights'''
        if True:
            weights_non_zero = []
            conv_channels_non_zero = []
            x_filters_non_zero = []
            for i, weight in enumerate(weights):
                if weight == 0:
                    pass
                else:
                    x_filters_non_zero.append(x_filters[i])
                    weights_non_zero.append(weight)
                    conv_channels_non_zero.append(conv_channels[i])
            assert len(x_filters_non_zero) == len(weights_non_zero)
            assert len(x_filters_non_zero) == len(conv_channels_non_zero)

        self.weights_non_zero = weights_non_zero

        ''' 2. Initialize 3d CONV layers'''
        if True:
            self.conv_layer_0 = nn.Conv3d(1, conv_channels_non_zero[0], kernel_size=2, bias=False)
            self.conv_layer_1 = nn.Conv3d(1, conv_channels_non_zero[1], kernel_size=2, bias=False)
            self.conv_layer_2 = nn.Conv3d(1, conv_channels_non_zero[2], kernel_size=2, bias=False)
            self.conv_layer_3 = nn.Conv3d(1, conv_channels_non_zero[3], kernel_size=2, bias=False)
            self.conv_layer_4 = nn.Conv3d(1, conv_channels_non_zero[4], kernel_size=2, bias=False)

            self.conv_layer_5 = nn.Conv3d(1, conv_channels_non_zero[5], kernel_size=2, bias=False)
            self.conv_layer_6 = nn.Conv3d(1, conv_channels_non_zero[6], kernel_size=2, bias=False)
            self.conv_layer_7 = nn.Conv3d(1, conv_channels_non_zero[7], kernel_size=2, bias=False)
            self.conv_layer_8 = nn.Conv3d(1, conv_channels_non_zero[8], kernel_size=2, bias=False)
            self.conv_layer_9 = nn.Conv3d(1, conv_channels_non_zero[9], kernel_size=2, bias=False)

            self.conv_layer_10 = nn.Conv3d(1, conv_channels_non_zero[10], kernel_size=2, bias=False)
            self.conv_layer_11 = nn.Conv3d(1, conv_channels_non_zero[11], kernel_size=2, bias=False)
            self.conv_layer_12 = nn.Conv3d(1, conv_channels_non_zero[12], kernel_size=2, bias=False)
            self.conv_layer_13 = nn.Conv3d(1, conv_channels_non_zero[13], kernel_size=2, bias=False)
            #
            # # self.conv_layer_14 = nn.Conv3d(1, conv_channels[14], kernel_size=2, bias=False)
            # # self.conv_layer_15 = nn.Conv3d(1, conv_channels[15], kernel_size=2, bias=False)
            # # self.conv_layer_16 = nn.Conv3d(1, conv_channels[16], kernel_size=2, bias=False)
            # # self.conv_layer_17 = nn.Conv3d(1, conv_channels[17], kernel_size=2, bias=False)
            # # self.conv_layer_18 = nn.Conv3d(1, conv_channels[18], kernel_size=2, bias=False)
            # # self.conv_layer_19 = nn.Conv3d(1, conv_channels[19], kernel_size=2, bias=False)

            self.conv_layers = [self.conv_layer_0, self.conv_layer_1, self.conv_layer_2, self.conv_layer_3, self.conv_layer_4,
                                self.conv_layer_5, self.conv_layer_6, self.conv_layer_7, self.conv_layer_8, self.conv_layer_9,
                                self.conv_layer_10, self.conv_layer_11, self.conv_layer_12, self.conv_layer_13, ]

                                # [self.conv_layer_14,
                                # self.conv_layer_15, self.conv_layer_16, self.conv_layer_17, self.conv_layer_18, self.conv_layer_19,
                                # ]
            assert len(self.conv_layers) == len(weights_non_zero)

        ''' 3. Initialize the weights of CONV layers'''
        if True:
            for i, conv_layer_i in enumerate(self.conv_layers):
                assert conv_layer_i.weight.shape[0] == x_filters_non_zero[i].shape[0]
                conv_layer_i.weight = nn.Parameter(torch.from_numpy(x_filters_non_zero[i] * 2 -1).float())

                # for param in conv_layer_i.parameters():
                #     param.requires_grad = False

        self.AvePool = nn.AvgPool2d(self.subpatch_size, stride=self.overlap_size)

    def ec_eq(self, num_patterns):
        ec = 0
        for i_str in num_patterns.keys():
            num_pattern = num_patterns[i_str]
            ec += self.weights_non_zero[int(i_str)] * num_pattern
        return ec

    def calculate_EC(self, x, isReturnEcMap, isReturnInternalFeatures, isSoft, isApproxBinary):
        a_features = {}        # [no_of_pattern_categories, ]f_hards
        b_features_ave_per_group = {}       # output_features
        c_EC = {}   # EC_hard_patches     torch.zeros(batch_size, 20).float().to(device)# in shape [batch, weight_n] or [number_equations, number_coefficients]
        d_EC_patches = {}      # pattern_no

        assert x.max() <= 1 and x.min() >= -1
        if isSoft and isApproxBinary:   # need to calculate EC twice
            x_binary = torch.ones_like(x)
            x_binary[x>=0] = 1

        for i, cnn_layer in enumerate(self.conv_layers):
            feature = cnn_layer(x)  # [batch, 128, ]
            if not isSoft:
                feature[feature != 8] = 0
                feature = feature / 8
            else:
                feature = feature / 16 + 0.5
                if isApproxBinary:  # only
                    feature_hard = cnn_layer(x_binary)
                    feature[feature_hard != 8] = 0

            a_features.update({str(i): feature})
            b_features_ave_per_group.update({str(i): feature.sum(dim=1)})
            c_EC.update({str(i): feature.sum(dim=(1, 2, 3, 4))})

            if isReturnEcMap:
                d_EC_patches.update({str(i): self.subpatch_size * self.subpatch_size * self.AvePool(feature.sum(dim=1))})

        EC = self.ec_eq(c_EC)
        result = {'EC': EC}

        if isReturnEcMap:
            EC_map = self.ec_eq(d_EC_patches)
            result.update({'EC_map': EC_map})

        if isReturnInternalFeatures:
            result.update({'EC_features': b_features_ave_per_group})
        return result

    def calculate_EC_step_by_step(self, x, isReturnEcMap, isReturnInternalFeatures, isSoft, isApproxBinary):
        a_features = []        # [no_of_pattern_categories, ]f_hards
        b_features_ave_per_group = []       # output_features
        c_EC = []   # EC_hard_patches     torch.zeros(batch_size, 20).float().to(device)# in shape [batch, weight_n] or [number_equations, number_coefficients]
        d_EC_patches = []      # pattern_no

        assert x.max() <= 1 and x.min() >= -1
        if isSoft and isApproxBinary:   # need to calculate EC twice
            x_binary = torch.ones_like(x)
            x_binary[x>=0] = 1

        for i, cnn_layer in enumerate(self.conv_layers):
            feature = cnn_layer(x)  # [batch, 128, ]
            if not isSoft:
                feature[feature != 8] = 0
                feature = feature / 8
            else:
                feature = feature / 16 + 0.5
                if isApproxBinary:  # only
                    feature_hard = cnn_layer(x_binary)
                    feature[feature_hard != 8] = 0

            a_features.append(feature)
            b_features_ave_per_group.append(feature.sum(dim=1))
            c_EC.append(feature.sum(dim=(1, 2, 3, 4)))

            if isReturnEcMap:
                d_EC_patches.append(self.subpatch_size * self.subpatch_size * self.AvePool(feature.sum(dim=1)))

        EC = self.ec_eq(c_EC)
        result = {'EC': EC}

        if isReturnEcMap:
            EC_map = self.ec_eq(d_EC_patches)
            result.update({'EC_map': EC_map})

        if isReturnInternalFeatures:
            result.update({'EC_features': b_features_ave_per_group})
        return result
    # def forward(self, x_hard, pad_value=-1, isReturnEcMap=True, isReturnInternalFeatures=False):
    def forward(self, x, pad_value=-1, isReturnEcMap=True, isReturnInternalFeatures=True, isSoft=False, isApproxBinary=False):
        '''return
        {
        'EC_hard'
        'features_hard'
        'EC_hard_patch'
        'EC_soft'
        'features_soft'
        'EC_soft_patch'
        }'''
        return_dist = {}
        p2d = (1,1,1,1,1,1)


        isApproxBinary = isApproxBinary and isSoft # isApproxBinary can = True only if isSoft =Ture, otherwise won't call isApproxBinary for an argmax {0,1} map
        # plt.figure(figsize=(4, 20))
        # plt.subplot(1, 2, 1).imshow(x_norm[0, 0, :, :].cpu().detach().numpy())
        # plt.subplot(1, 2, 2).imshow(x_norm_pad[0, 0, :, :].cpu().detach().numpy())
        # # plt.subplot(1, 4, 3).imshow(f1[0, 2, :, :].cpu().detach().numpy())
        # # plt.subplot(1, 4, 4).imshow(f1[0, 3, :, :].cpu().detach().numpy())
        # plt.show()
        #
        # plt.figure(figsize=(80, 20))
        # plt.subplot(1, 4, 1).imshow(f4[0, 0, :, :].cpu().detach().numpy())
        # plt.subplot(1, 4, 2).imshow(f4[0, 1, :, :].cpu().detach().numpy())
        # plt.subplot(1, 4, 3).imshow(f4[0, 2, :, :].cpu().detach().numpy())
        # plt.subplot(1, 4, 4).imshow(f4[0, 3, :, :].cpu().detach().numpy())
        # plt.show()

        '''EC hard'''
        x_norm = x * 2 -1         # either continuous \in [-1, 1] or binary \in {-1, 1}
        x_norm_pad = nn.functional.pad(x_norm, p2d, 'constant', pad_value)

        result = self.calculate_EC(x_norm_pad, isReturnEcMap, isReturnInternalFeatures, isSoft, isApproxBinary)

        return result

