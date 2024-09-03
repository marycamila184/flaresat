import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        
        bce_loss = nn.functional.binary_cross_entropy(probs, targets, reduction='none')
        
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = alpha_factor * ((1 - p_t) ** self.gamma)
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss