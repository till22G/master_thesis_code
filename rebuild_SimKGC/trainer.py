import torch.utils.data

from model import build_model
from data_structures import Dataset

class CustomTrainer:
    def __init__(self, args) -> None:
        build_model(args)
        
        # load datasets
        train_dataset = Dataset(args.train_path)
        valid_dataset = Dataset(args.valid_path)
        
        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True
        )
        
        self.valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True
        )