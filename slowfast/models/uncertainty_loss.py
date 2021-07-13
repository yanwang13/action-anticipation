import torch
import torch.nn as nn
import numpy as np

"""
This is implementation is referenced from
https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
"""
class Uncertaintyloss(nn.Module):
    def __init__(self, num_tasks=2):
        super(Uncertaintyloss, self).__init__()

        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros((self.num_tasks)))

    def forward(self, preds, labels, mode=2):
        loss = 0
        #for idx in range(self.num_tasks):
        #    precision = torch.exp(-self.log_vars[idx])
        #    loss_ce = nn.CrossEntropyLoss()(preds[idx], labels[idx])
        #    loss += torch.sum(loss_ce*precision + self.log_vars[idx], -1)

        verb_precision = torch.exp(-self.log_vars[0])
        loss_verb = nn.CrossEntropyLoss(reduction='none')(preds[0], labels[0])
        loss += (loss_verb*verb_precision + self.log_vars[0])

        noun_precision = torch.exp(-self.log_vars[1])
        loss_noun = nn.CrossEntropyLoss(reduction='none')(preds[1], labels[1])
        loss += (loss_noun*noun_precision + self.log_vars[1])

        return torch.mean(loss), torch.mean(loss_verb), torch.mean(loss_noun)

    def get_weights(self):
        verb_weight = np.exp(-self.log_vars[0].item())
        noun_weight = np.exp(-self.log_vars[1].item())
        return verb_weight, noun_weight
