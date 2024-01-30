# this file should contain the model of SimKGC recreated from the description in the paper
import argparse
import torch
import os

from transformers import AutoModel, AutoTokenizer

from argparser import args
from logger import logger
from my_trainer import CustomTrainer
#from trainer import CustomTrainer

def main():
    logger.info("Entered Main")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # change directory, so the script will find the data
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath) 
    os.chdir(dname)

    trainer = CustomTrainer(args)
    trainer.training_loop()


if __name__ == "__main__":
    main()