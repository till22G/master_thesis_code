import torch

from typing import List


def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_accuracy(logits: torch.tensor, labels: torch.tensor, topk=(1,)) -> List[torch.tensor]:
    with torch.no_grad():
        max_k = max(topk)
        _ , idx_topk_pred = logits.topk(max_k, dim=1)
        labels = labels.unsqueeze(dim=1).expand(idx_topk_pred.shape)
        correct_classified = (labels == idx_topk_pred)
        accuracies = []
        for k in topk:
            acc = (correct_classified[:, :k].sum() / len(labels)) * 100
            accuracies.append(acc)
            
    return accuracies