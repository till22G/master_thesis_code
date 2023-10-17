import torch

def move_to_cuda(data): 
    if len(data) == 0: return {}
    
    def _move_to_cuda(data):
        if torch.is_tensor(data): return data.cuda(non_blocking=True)
        if isinstance(data, dict): return {key: _move_to_cuda(value) for key, value in data.items()}
        if isinstance(data, tuple): return (_move_to_cuda(value) for value in data)
        if isinstance(data, list): return [_move_to_cuda(item) for item in data]
        else: return data
    
    return _move_to_cuda(data)