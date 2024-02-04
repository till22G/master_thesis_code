import torch.utils.data
import torch.nn as nn

from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from model import build_model
from data_structures import Dataset, collate_fn
from logger import logger
from help_functions import move_to_cuda, calculate_accuracy, calculate_running_mean, save_checkpoints

class CustomTrainer:
    def __init__(self, args) -> None:
        
        self.args = args
        self.model = build_model(self.args)
        logger.info(self.model)
        
        # use GPUs if possible
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
            logger.info("{} cuda devices found. GPUs will be used".format(torch.cuda.device_count()))
        elif torch.cuda.device_count() == 1:
            self.model.cuda()
            logger.info("{} cuda device found. GPU will be used".format(torch.cuda.device_count()))
        else: 
            logger.info("No GPU available. CPU will be used")
          
        # construct GradScaler when using amp  
        if self.args.use_amp: self.scaler = torch.cuda.amp.GradScaler()    
            
        # get criterion and optimizer
        self.best_metric = None
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
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=args.warmup,
                                                            num_training_steps=num_training_steps)
        
        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            num_workers=self.args.num_workers,
            prefetch_factor=10
        )
        
        self.valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            num_workers=self.args.num_workers
        )
        
    def training_loop(self):
        
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
        for epoch in range(1, self.args.num_epochs + 1):
            self.trian_epoch(epoch)
            self.evaluate_epoch(epoch)
            
        logger.info("Finished training")
    
    def trian_epoch(self, epoch):
        
        logger.info("Starting training epoch {}/{}".format(epoch, self.args.num_epochs))
        
        # enumarate over taining data
        for i, batch_dict in enumerate(tqdm(self.train_data_loader)):
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


            model = self.model.module if hasattr(self.model, "module") else self.model
            model_output = model.compute_logits(encodings=model_output, batch_data=batch_dict)
            logits, labels = model_output.get("logits"), model_output.get("labels")
            
            self.optimizer.zero_grad()
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self.criterion(logits, labels)
                    # they also included loss for tails -> head + relation
                    loss += self.criterion(logits[:, :self.args.batch_size].t(), labels)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                loss = self.criterion(logits, labels)
                loss += self.criterion(logits[:, :self.args.batch_size].t(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                
            self.lr_scheduler.step()

            count = 0
            if (i + 1) % 5000 == 0:
                count += 100
                # save model checkpoint
                save_dict = {"state_dict" : self.model.state_dict(),
                            "args" : self.args.__dict__,
                            "epoch" : epoch}
                save_checkpoints(self.args, save_dict, count + epoch)
                
    @torch.no_grad()        
    def evaluate_epoch(self, epoch):
        if not self.valid_data_loader:
            return {}
        
        logger.info("Starting evaluation epoch")
        
        runnig_mean_acc = []
        running_mean_loss = 0.0
        
        for i, batch_dict in enumerate(tqdm(self.valid_data_loader)):
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            # set model in evaluation mode
            self.model.eval()
            
            model_output = self.model(**batch_dict)
            model = self.model.module if hasattr(self.model, "module") else self.model
            model_output = model.compute_logits(encodings=model_output , batch_data=batch_dict)
            logits, labels = model_output.get("logits"), model_output.get("labels")
            
            # calculate loss
            loss = self.criterion(logits, labels)
            running_mean_loss = calculate_running_mean(running_mean_loss, loss, i+1)
           
            # calculate accuracy
            accuracy = calculate_accuracy(logits, labels, topk=(1, 3))
            runnig_mean_acc = calculate_running_mean(runnig_mean_acc, accuracy, i+1)
            
        # report values
        logger.info("Epoch {}/{}: mean loss: {:.5f}, top 1 accuracy: {:.5f}, top 3 accuracy: {:.5f}"
                    .format(epoch, self.args.num_epochs, running_mean_loss, 
                            runnig_mean_acc[0], runnig_mean_acc[1]))
        
        
        # save model checkpoint
        save_dict = {"state_dict" : self.model.state_dict(),
                     "args" : self.args.__dict__,
                     "epoch" : epoch}
        
        is_best = self.best_metric is None or runnig_mean_acc[0] > self.best_metric
        if is_best: self.best_metric = runnig_mean_acc[0]
        save_checkpoints(self.args, save_dict, epoch, is_best)
        
