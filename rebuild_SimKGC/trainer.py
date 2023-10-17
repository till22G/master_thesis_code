import torch.utils.data
import torch.nn as nn
import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup

from model import build_model
from data_structures import Dataset, collate_fn
from logger import logger

class CustomTrainer:
    def __init__(self, args) -> None:
        
        self.args = args
        self.model = build_model(self.args)
        logger.info(self.model)
        
        # use GPUs if possible
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
            logger.info("{} cuda devices found. GPUs will be used".format(torch.cuda.device_count()))
        elif torch.cuda.device_count() == 1:
            self.model.cuda()
            logger.info("{} cuda device found. GPU will be used".format(torch.cuda.device_count()))
        else: 
            logger.info("No GPU available. CPU will be used")
            
        # get criterion and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)
        
        # load datasets
        train_dataset = Dataset(self.args.train_path)
        valid_dataset = Dataset(self.args.valid_path)
        
        # get parameters
        assert self.args.batch_size > 0, "Batch size must be larger than 0"
        num_training_steps = self.args.num_epochs * len(train_dataset) // self.args.batch_size
        args.warmup = min(self.args.warmup, num_training_steps // 10)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                       num_warmup_steps=args.warmup,
                                                       num_training_steps=num_training_steps)
        
        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        self.valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.args.batch_size * 2,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
    def training_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
        for epoch in tqdm(range(self.args.epochs)):
            self.tarin_epoch(epoch)
            self.elvauate_epoch(epoch)
    
    def evaluation_loop(self):
        pass
    
    def trian_epoch(self, epoch):
        for i, batch_dict in enumerate(self.train_loader):
            if torch.cuda.is_available():
                
        
    def evlauate_epoch():
        pass
    