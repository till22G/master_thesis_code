import torch.utils.data
import torch.nn as nn
import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup

from model import build_model
from data_structures import Dataset, collate_fn
from logger import logger
from help_functions import move_to_cuda

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
          
        # construct GradScaler when using amp  
        if self.args.use_amp: self.scaler = torch.cuda.amp.GradScaler()    
            
        # get criterion and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
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
            
        for epoch in range(self.args.num_epochs):
            self.trian_epoch(epoch)
            self.evaluate_epoch(epoch)
    
    def evaluation_loop(self):
        pass
    
    def trian_epoch(self, epoch):
        
        # enumarate over taining data
        for i, batch_dict in enumerate(self.train_data_loader):
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)

            # set model in train mode
            self.model.train()
            
            # compute encodings and logits
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    model_output = self.model(**batch_dict)
            else:
                model_output = self.model(**batch_dict)

            model = model.module if hasattr(self.model, "module") else self.model
            model_output = self.model.compute_logits(encodings=model_output , batch_data=batch_dict)
            logits, labels = model_output.get("logits"), model_output.get("labels")
            
            loss = self.criterion(logits, labels)
            # they also included loss for tails -> head + relation
            
            self.optimizer.zero_grad()
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    loss.backward()
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                
            self.lr_scheduler.step()

            
            logger.info("{}/{}".format(i, len(self.train_data_loader)))  
        logger.info("Epoch {}/{}".format(epoch, self.args.num_epochs))
        
                
    def evaluate_epoch():
        pass