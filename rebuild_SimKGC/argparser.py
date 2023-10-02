import argparse

parser = argparse.ArgumentParser(prog="replicated SimKGC",
                                 description="Argparser for replicating SimKGC")

parser.add_argument("--pretrained-model", default="bert-base-uncased", type=str, 
                    help="define the tranformer model that is used for encoder initialization")
parser.add_argument("--test-mode", action="store_true",
                    help="Start SimKGC in test mode or not")
parser.add_argument("--task", default="fb15k237", type=str,
                    help="Select the dataset that is used for training and validation")
parser.add_argument("--train-path", default="", type=str,
                    help="Define the path to the training data")
parser.add_argument("--valid-path", default="", type=str,
                    help="Define the path to the validation data")
parser.add_argument("--max-number-tokens", default=50, type=int,
                    help="Specify the maximal number of tokens returned by the tokenizer")

args = parser.parse_args()