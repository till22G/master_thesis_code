import torch.utils.data

from model import build_model
from data_structures import Dataset, collate_fn
from logger import logger

class CustomTrainer:
    def __init__(self, args) -> None:
        
        self.args = args
        self.model = build_model(self.args)
        logger.info(self.model)
        
        # use gpus if possble
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            logger.info("{} cuda devices found. GPUs will be used")
        elif torch.cuda.device_count() == 1:
            self.model.cuda()
            logger.info("{} cuda devices found. GPU will be used")
        else: 
            logger.info("No GPU available. CPU will be used")
        
        # load datasets
        train_dataset = Dataset(args.train_path)
        valid_dataset = Dataset(args.valid_path)
        
        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )