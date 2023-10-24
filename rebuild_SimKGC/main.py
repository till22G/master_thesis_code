# this file should contain the model of SimKGC recreated from the description in the paper
import argparse
import torch
import os

from transformers import AutoModel, AutoTokenizer

from argparser import args
from logger import logger
from trainer import CustomTrainer

def main():
    logger.info("Entered Main")
    
    # change directory, so the script will find the data
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath) 
    os.chdir(dname)
    
    args.train_path = "../data/fb15k237/train.json"
    args.valid_path = "../data/fb15k237/valid.json"
    trainer = CustomTrainer(args)
    trainer.training_loop()


if __name__ == "__main__":
    main()