import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable






# Entropy based Margin-Loss
class EMLoss(nn.Module):

    def __init__(self, beta=0.4, margin=0.4):
        super(EMLoss, self).__init__()

        self.beta = beta
        self.margin = margin



    def forward(self, id_input, ood_input, target):
        cross_entropy = - torch.sum( torch.log(id_input) * target ) / id_input.shape[0]

        id_entropy = -torch.sum( id_input*torch.log(id_input) ) / id_input.shape[0]

        ood_entropy = -torch.sum( ood_input*torch.log(ood_input) ) / ood_input.shape[0]

        margin_diff = self.beta * max(self.margin + id_entropy - ood_entropy, 0)



        return cross_entropy + margin_diff