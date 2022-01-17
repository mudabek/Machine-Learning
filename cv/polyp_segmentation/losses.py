import torch.nn as nn
import torch.nn.functional as F
import torch



class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


class CombinedLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CombinedLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()
        # self.logcosh = LogCoshDiceLoss()

        # self.bce_coeff = 1.0
        # self.dice_coeff = 1.0
        # self.logcosh_coeff = 1.0

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        # logcoshloss = self.logcosh(pred, target)

        loss = diceloss + bceloss

        return loss