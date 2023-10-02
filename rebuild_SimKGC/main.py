# this file should contain the model of SimKGC recreated from the description in the paper
import argparse
import torch

from transformers import AutoModel, AutoTokenizer

from argparser import args
from logger import logger
from trainer import CustomTrainer

def main():
    logger.info("Entered Main")
    
    trainer = CustomTrainer(args)

if __name__ == "__main__":
    main()