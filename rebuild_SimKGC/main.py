# this file should contain the model of SimKGC recreated from the description in the paper
import argparse
import torch

from transformers import AutoModel, AutoTokenizer

from logger import logger

"""  load the bert-base-uncased model
model_type = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_type)
bert_hr = AutoModel.from_pretrained(model_type) # create bert model for relation aware embeddings
bert_t = AutoModel.from_pretrained(model_type) # create bert model for tail entity embeddings

input_text = "I love this library!"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = bert_hr(**inputs)

hidden_states = outputs.last_hidden_state

print(hidden_states)
print(hidden_states.size())
#last_layer = model.base_model.encoder.layer[-1] """

def main():
    logger.info("Test info")

if __name__ == "__main__":
    main()