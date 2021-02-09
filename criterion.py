import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

# class Criterion(_Loss):
#     def __init__(self):
#         super(Criterion, self).__init__()
#         # self.amount = way * shot

#     def forward(self, probs, target):  # (Q,C) (Q)
#         # target = target[self.amount:]
#         target_onehot = torch.zeros_like(probs)
#         target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
#         loss = torch.mean((probs - target_onehot) ** 2)
#         pred = torch.argmax(probs, dim=1)
#         acc = torch.sum(target == pred).float() / target.shape[0]
#         return loss, acc

class Criterion(_Loss):
    def __init__(self):
        super(Criterion, self).__init__()
        # self.amount = way * shot

    def log_softmax(self, x):
        return x - torch.logsumexp(x,dim=1, keepdim=True)


    def forward(self, output, target):  # (Q,C) (Q)
        # target = target[self.amount:]
        # target_onehot = torch.zeros_like(output, dtype=torch.long)
        # target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)

        num_examples = target.shape[0]
        batch_size = output.shape[0]
        output = self.log_softmax(output)
        output = output[range(batch_size), target]
        return - torch.sum(output)/num_examples
        
        
        # loss = torch.mean((probs - target_onehot) ** 2)
        # pred = torch.argmax(probs, dim=1)
        # acc = torch.sum(target == pred).float() / target.shape[0]
        # return loss, acc

# https://byeongjokim.github.io/posts/Cosine-Loss/
class CosineLoss(_Loss):
    def __init__(self, device, xent=.1, reduction="sum"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        
        self.y = torch.Tensor([1]).to(device).detach()
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)
        
        return cosine_loss + self.xent * cent_loss

# Unused########
def log_softmax(x):
    return x - torch.logsumexp(x,dim=1, keepdim=True)

def cross_etrp(output, target):  # (Q,C) (Q)
    # target = target[self.amount:]
    # target_onehot = torch.zeros_like(output, dtype=torch.long)
    # target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)

    num_examples = target.shape[0]
    batch_size = output.shape[0]
    output = log_softmax(output)
    output = output[range(batch_size), target]
    return - torch.sum(output)/num_examples

#################