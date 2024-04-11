import torch
import torch.nn.functional as F

import torchmetrics

class F1Score(torchmetrics.Metric):
    higher_is_better = True

    def __init__(
            self, top_k=1, **kwargs):
        super().__init__(**kwargs)

        self.top_k = top_k

        self.add_state('tp', default=torch.tensor(0.0, dtype=torch.float32),
                       dist_reduce_fx='sum')
        self.add_state('fpfn', default=torch.tensor(0.0, dtype=torch.float32),
                       dist_reduce_fx='sum')


    def update(self, preds, target):
        if self.top_k == 1:
            preds = torch.argmax(preds, dim=-1)

            self.tp += torch.sum(preds == target)
            self.fpfn += torch.sum(preds != target)

        else:
            target = torch.unsqueeze(target, dim=-1)
            preds = torch.topk(preds, k=self.top_k).indices

            self.tp += torch.sum((target == preds).any(dim=-1))
            self.fpfn += torch.sum(torch.all(target != preds, dim=-1))


    def compute(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fpfn)
