import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricEstimator(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.l0 = nn.Linear(in_feature*2, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        # self.apply(init_weights)

    def forward(self, visual, language):

        x = torch.cat((visual, language), dim=-1)
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out
    
class SymmetricEstimator_silu(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.l0 = nn.Linear(in_feature*2, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        # self.apply(init_weights)

    def forward(self, visual, language):

        x = torch.cat((visual, language), dim=-1)
        out = F.silu(self.l0(x))
        out = F.silu(self.l1(out))
        out = self.l2(out)
        return out
    
class SymmetricResEstimator(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.l0 = nn.Linear(in_feature*3, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        # self.apply(init_weights)

    def forward(self, visual, language):

        x = torch.cat((visual, language, torch.abs(visual - language)), dim=-1)
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out
    
class SimpleParallelEstimator(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.desc_net = SymmetricEstimator(in_feature)
        self.motion_net = SymmetricEstimator(in_feature)

    def forward(self, visual, language):
        desc_lang, motion_lang = language.chunk(2, dim=-1)
        desc_out = self.desc_net(visual, desc_lang)
        motion_out = self.motion_net(visual, motion_lang)
        return desc_out + motion_out