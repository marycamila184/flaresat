import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss