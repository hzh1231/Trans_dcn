import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels):
        super(PAM_Module, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        #创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        b,c,h,w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h*w).transpose(1, 2), C.view(b, c, h*w)))
        E = torch.matmul(D.view(b, c, h*w), S.transpose(1, 2)).view(b,c,h,w)
        #gamma is a parameter which can be training and iter
        E = self.gamma * E + x
        return E
        

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)


    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        b,c,h,w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h*w)).view(b, c, h, w)
        X = self.beta * X + x
        return X