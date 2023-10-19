import torch

from typing import List

def move_to_cuda(data): 
    if len(data) == 0: return {}
    
    def _move_to_cuda(data):
        if torch.is_tensor(data): return data.cuda(non_blocking=True)
        if isinstance(data, dict): return {key: _move_to_cuda(value) for key, value in data.items()}
        if isinstance(data, tuple): return (_move_to_cuda(value) for value in data)
        if isinstance(data, list): return [_move_to_cuda(item) for item in data]
        else: return data
    
    return _move_to_cuda(data)


def calculate_accuracy(logits: torch.tensor, labels: torch.tensor, topk=(1, )) -> List[torch.tensor]:
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

def calculate_running_mean(runnig_mean, new_datapoint, iteration: int):
    
    if isinstance(runnig_mean, list):
        for i in range(len(runnig_mean)):
            tmp = (new_datapoint[i] - runnig_mean[i]) / iteration
            runnig_mean[i] = runnig_mean[i] + tmp
        return runnig_mean
    
    elif isinstance(runnig_mean, float) or isinstance(runnig_mean, int):
        tmp = (new_datapoint - runnig_mean) / iteration
        runnig_mean = runnig_mean + tmp
        return runnig_mean