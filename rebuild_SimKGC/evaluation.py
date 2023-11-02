import torch
import os
import json
import torch.utils.data.dataset

from tqdm import tqdm
from typing import Optional, List

from data_structures import Dataset, collate_fn
from argparser import args
from logger import logger
from triplet_mask import construct_triplet_mask, construct_self_negative_mask
from help_functions import move_to_cuda
from model import build_model

args.train_path = "../data/fb15k237/"
args.task = "fb15k237"

args.pre_batch = False
args.use_self_negative = False

model = build_model(args)
logger.info(model)
ckt_path = "../model_checkpoints/fb15k237/best_model_checkpoint.mdl"
ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)


# DataParallel will introduce 'module.' prefix
state_dict = ckt_dict["model_state_dict"]

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}


model.load_state_dict(state_dict, strict=True)

# use GPUs if possible
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    logger.info("{} cuda devices found. GPUs will be used".format(torch.cuda.device_count()))
elif torch.cuda.device_count() == 1:
    model.cuda()
    logger.info("{} cuda device found. GPU will be used".format(torch.cuda.device_count()))
else: 
    logger.info("No GPU available. CPU will be used")


# load test set
test_dataset = Dataset("../data/fb15k237/test.json")

test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            num_workers=args.num_workers
        )


for i, batch_dict in enumerate(tqdm(test_data_loader)):
    if torch.cuda.is_available():
        batch_dict = move_to_cuda(batch_dict)
        
        # set model in evaluation mode
        model.eval()
        
        model_output = model(**batch_dict)
        model = model.module if hasattr(model, "module") else model
        model_output = model.compute_logits(encodings=model_output , batch_data=batch_dict)
        logits, labels = model_output.get("logits"), model_output.get("labels")
        
        # calculate loss
        loss = criterion(logits, labels)
        running_mean_loss = calculate_running_mean(running_mean_loss, loss, i+1)
        
        # calculate accuracy
        accuracy = calculate_accuracy(logits, labels, topk=(1, 3))
        runnig_mean_acc = calculate_running_mean(runnig_mean_acc, accuracy, i+1)
        
    # report values
        logger.info("Epoch {}/{}: mean loss: {:.5f}, top 1 accuracy: {:.5f}, top 3 accuracy: {:.5f}"
                    .format(epoch, args.num_epochs, running_mean_loss, 
                            runnig_mean_acc[0], runnig_mean_acc[1]))