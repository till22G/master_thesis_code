# this file should contain the model of SimKGC recreated from the description in the paper
import argparse
import torch

from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input_text = "I love using the Transformers library!"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state

print(hidden_states)
#last_layer = model.base_model.encoder.layer[-1]

