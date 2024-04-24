import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
##zf shoudle CIF torch.cat    xf shoudle DFF

class RGBT_fusion(nn.Module):

    def __init__(self, channel=256,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        ##full connection layer
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()  ##0----1 value
        )
        self.cv=nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
    def forward(self, rgb_feature, t_feature):
        ##for rgb
        b, c, w, h = rgb_feature.size()
        y_r = self.avg_pool(rgb_feature).view(b, c)
        weigh_rgb = self.fc(y_r).view(b, c, 1, 1)
        ##for t
        b, c, w, h = t_feature.size()
        y_t = self.avg_pool(t_feature).view(b, c)
        weigh_t = self.fc(y_t).view(b, c, 1, 1)
        # union_feature = torch.cat((rgb_feature,t_feature),1)
        # b,c,w,h = union_feature.size()
        # y = self.avg_pool(union_feature).view(b, c)
        # weigh= self.fc(y).view(b, c, 1, 1)
        # weigh_rgb,weigh_t=torch.chunk(weigh,2,dim=1)
        fea_r=rgb_feature+(t_feature*weigh_t)
        fea_t=t_feature+(rgb_feature*weigh_rgb)
        fea_rt=torch.cat((fea_r,fea_t),1)
        channel_num = fea_rt.shape[1]
        feature_fusion=self.cv(fea_rt)

        return feature_fusion



if __name__ == '__main__':
    rgb_feature=torch.randint(1,100,(3,256,7,7))
    t_feature=torch.randint(1,100,(3,256,7,7))
    rgb_feature=rgb_feature.cuda()
    t_feature=t_feature.cuda()
    print(rgb_feature)
    new_rgb_feature=RGBT_fusion(rgb_feature,t_feature)
    print(new_rgb_feature)