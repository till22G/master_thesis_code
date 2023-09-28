import argparse

parser = argparse.ArgumentParser(prog="replicated SimKGC",
                                 description="Argparser for replicating SimKGC")

parser.add_argument("--pretrained-model", default="bert-base-uncased", 
                    type=str, help="define the tranformer model that is used for encoder initialization")
parser.add_argument("--test-mode", action="store_true",
                    help="Start SimKGC in test mode or not")