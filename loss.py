import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def entropy(p):
    return -torch.sum( p * torch.log(p) )


# Entropy based Margin-Loss
class EMLoss(nn.Module):

    def __init__(self, beta=0.4, margin=0.4):
        super(EMLoss, self).__init__()

        self.beta = beta
        self.margin = margin



    def forward(self, id_input, ood_input, target):
        cross_entropy = -torch.sum( torch.log(id_input) * target ) / id_input.shape[0]

        id_entropy = entropy(id_input) / id_input.shape[0]

        ood_entropy = entropy(ood_input) / ood_input.shape[0]

        margin_diff = self.beta * max(self.margin + id_entropy - ood_entropy, 0)



        return cross_entropy + margin_diff


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, input_data):
        return entropy(input_data)