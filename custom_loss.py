import torch
from torch import nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    # You should try with temperature latter
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature
    
    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            # negative samples with higher scores should be punished more, have larger weights. 
            weights= F.softmax(self.adv_temperature * n_scores, dim=-1).detach() # Using detach here because according to the paper, these probabilities depend 
                                                            # only on the current state of embeddings. Shouldn't be optimized
            n_scores = weights * n_scores
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        
        return (p_loss + n_loss) / 2, p_loss, n_loss 
