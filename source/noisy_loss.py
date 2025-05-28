import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        softmax_probs = F.softmax(logits, dim=1)
        confidence_in_target = softmax_probs[torch.arange(len(targets)), targets]
        weights = (1 - self.p) + self.p * confidence_in_target
        return (losses * weights).mean()