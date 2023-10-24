import argparse

parser = argparse.ArgumentParser(prog="replicated SimKGC",
                                 description="Argparser for replicating SimKGC")

parser.add_argument("--pretrained-model", default="bert-base-uncased", type=str, 
                    help="define the transformer model that is used for encoder initialization")
parser.add_argument("--task", default="fb15k237", type=str,
                    help="Select the dataset that is used for training and validation")
parser.add_argument("--train-path", default="", type=str,
                    help="Define the path to the training data")
parser.add_argument("--valid-path", default="", type=str,
                    help="Define the path to the validation data")
parser.add_argument("--max-number-tokens", default=50, type=int,
                    help="Specify the maximal number of tokens returned by the tokenizer")
parser.add_argument("--use-neighbors", action="store_true",
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
parser.add_argument("--batch-size", default=8, type=int,
                    help="set the mini-batch size for training")
parser.add_argument("--pre-batch", default=2, type=int,
                    help="number of pre-batches used to compute pre-batch negatives")
parser.add_argument("--use-inverse-triples", action="store_true",
                    help="specify whether inverse triple should be loaded to the data or not")
parser.add_argument("--use-self-negatives", action="store_true",
                    help="toggle whether self negatives are used for training or not")
parser.add_argument("--pre-batch-weight", default=0.5, type=float,
                    help="set the weight of the pre-batch logits")
parser.add_argument("--learning-rate", default=2e-5, type=float,
                    help="set the initial learning rate for the training")
parser.add_argument("--weight-decay",default=1e-4,type=float,
                    help="set the weight decay for the learning rate during training")
parser.add_argument("--num-epochs", default=10, type=int,
                    help="set the number of epochs for training")
parser.add_argument("--warmup", default=400, type=int,
                    help="set the number of warmup steps for the training")
parser.add_argument("--use-amp", action="store_true",
                    help="specify whether amp should be used or not")
parser.add_argument("--grad-clip", default=10, type=float,
                    help="define value for gradient clip")
parser.add_argument("--model-dir", default="test_model_dict", type=str,
                    help="define the output diretory for the model")
args = parser.parse_args()