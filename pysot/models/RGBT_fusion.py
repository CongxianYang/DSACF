import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
##zf shoudle CIF torch.cat    xf shoudle DFF

class RGBT_fusion(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        ##full connection layer
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb_feature, t_feature):
        channel_num = rgb_feature.shape[1]
        union_feature = torch.cat((rgb_feature,t_feature),1)
        b,c,w,h = union_feature.size()
        y = self.avg_pool(union_feature).view(b, c)
        weigh= self.fc(y).view(b, c, 1, 1)
        weigh_rgb,weigh_t=torch.chunk(weigh,2,dim=1)
        feature_fusion=rgb_feature+t_feature*weigh_t+t_feature+rgb_feature*weigh_rgb
        return feature_fusion



if __name__ == '__main__':
    rgb_feature=torch.randint(1,100,(3,256,7,7))
    t_feature=torch.randint(1,100,(3,256,7,7))
    rgb_feature=rgb_feature.cuda()
    t_feature=t_feature.cuda()
    print(rgb_feature)
    new_rgb_feature=RGBT_fusion(rgb_feature,t_feature)
    print(new_rgb_feature)