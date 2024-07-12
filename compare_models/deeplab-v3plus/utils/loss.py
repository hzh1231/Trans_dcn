import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dice_loss import DiceLoss

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'multi':
            return self.MultiLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        '''
        n, c, h, w = logit.size()
        target_one_hot = F.one_hot(target.long(), c)
        target = target_one_hot.permute(0, 3, 1, 2)
        BCE_loss = F.binary_cross_entropy_with_logits(logit, target.float(), reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        loss = torch.mean(F_loss)
        '''

        if self.batch_average:
            loss /= n

        return loss

    def MultiLoss(self, logit, target):
        '''
        # Focal Loss
        gamma = 2
        alpha = 0.5
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        Focal_loss = -((1 - pt) ** gamma) * logpt
        '''
        
        # CE Loss
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        CE_loss = criterion(logit, target.long())

        # Dice Loss
        num = target.size(0)
        smooth = 1.
        eps = 1e-8
        probs = F.sigmoid(logit)
        # target = torch.unsqueeze(target, 1)
        # target = target.repeat(1, 2, 1, 1)
        target_one_hot = F.one_hot(target.long(), c)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)
        m1 = probs.view(num, -1)
        m2 = target_one_hot.contiguous().view(num, -1)
        intersection = m1 * m2
        Dice_loss = 1 - (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth + eps)
        Dice_loss = Dice_loss.mean()

        # dice_loss = DiceLoss()
        # Dice_loss = dice_loss.dice(logit, target.long())

        total_loss = (0.5 * CE_loss) + (0.5 * Dice_loss)

        if self.batch_average:
            total_loss /= n

        return total_loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




