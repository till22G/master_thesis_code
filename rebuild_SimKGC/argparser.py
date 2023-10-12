import argparse

parser = argparse.ArgumentParser(prog="replicated SimKGC",
                                 description="Argparser for replicating SimKGC")

parser.add_argument("--pretrained-model", default="bert-base-uncased", type=str, 
                    help="define the transformer model that is used for encoder initialization")
parser.add_argument("--test-mode", action="store_true",
                    help="Start SimKGC in test mode or not")
parser.add_argument("--task", default="fb15k237", type=str,
                    help="Select the dataset that is used for training and validation")
parser.add_argument("--train-path", default="train.json", type=str,
                    help="Define the path to the training data")
parser.add_argument("--valid-path", default="", type=str,
                    help="Define the path to the validation data")
parser.add_argument("--max-number-tokens", default=50, type=int,
                    help="Specify the maximal number of tokens returned by the tokenizer")
parser.add_argument("--use-neighbors", action="store_false",
                    help="Set whether context from neighboring nodes should be used when \
                    the descriptions are short (< 20 tokens)")
parser.add_argument("--use-descriptions", action="store_true",
                    help="Use the entity descriptions in the encodings")
parser.add_argument("--t", default=0.05, type=float, 
                    help="temperature parameter for loss function")
parser.add_argument("--finetune-t", action="store_true",
                    help="set whether t should be finetuned during training or not")
parser.add_argument("--additive-margin", default=0.02, type=float,
                    help="set additive margin in InfoNCE loss function")
parser.add_argument("--batch-size", default=1024, type=int,
                    help="set the mini-batch size for training")
parser.add_argument("--pre-batch", default=0, type=int,
                    help="number of pre-batches used to compute pre-batch negatives")
parser.add_argument("--use-inverse-triples", action="store_false",
                    help="specify whether inverse triple should be loaded to the data or not")
args = parser.parse_args()