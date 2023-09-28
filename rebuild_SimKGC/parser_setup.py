import argparse

parser = argparse.ArgumentParser(prog="replicated SimKGC",
                                 description="Argparser for replicating SimKGC")

parser.add_argument("--pretrained-model", default="bert-base-uncased", type=str)