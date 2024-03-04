## Code Repository for Master Thesis Tillman Galla

## Technical Requirements
* python>=3.7
* torch>=1.6
* transformers>=4.15


The requiremnts file for the environment used during experiments can be found here: [requirements.txt](requirements.txt)

The code has been implemented to run on CUDA devices. A implementation for CPUs has been omitted, so please make sure there are 
CUDA devices available when runnig the code.

Most experiments have been conducted using 3 RTX A6000 (48GB) GPUs.

## When encountering "CUDA out of memory" error:
    1. Make sure there are enough CUDA devices available
    2. Reduce batch size (impacts performance due to in-batch negative loss)

Following the original implementation data for WN18RR and FB15k-237 is taken from [KG-BERT](https://github.com/yao8839836/kg-bert).

Wikidata5M can be downloaded with this script taken from: https://github.com/intfloat/SimKGC/blob/main/scripts/download_wikidata5m.sh
Just run this line:

```
bash scripts/download_wikidata5m.sh
```

### WN18RR

#### Preprocessing

For data preprocessing run:
```
bash scripts/preprocess.sh WN18RR
```

#### Training

For model training run:
```
OUTPUT_DIR=/output/dir bash scripts/train_wn18rr.sh 
```

#### Evaluation

For evaluation run:
```
bash scripts/eval.sh /path/to/model/ WN18RR 
```


### FB15k-237

#### Preprocessing

```
bash scripts/preprocess.sh FB15k237
```

For model training run:
```
OUTPUT_DIR=/output/dir bash scripts/train_fb.sh
```


For evaluation run:
```
bash scripts/eval.sh /path/to/model/ FB15k237 
```


### Wikidata5M-Trans

#### Preprocessing

For data preprocessing run:
```
bash scripts/preprocess.sh wiki5m_trans
```

For model training run:
```
OUTPUT_DIR=/output/dir bash scripts/train_wiki.sh wiki5m_trans
```

For evaluation run:
```
bash scripts/eval.sh /path/to/model/ wiki5m_trans 
```

### Wikidata5M-Ind

#### Preprocessing

For data preprocessing run:
```
bash scripts/preprocess.sh wiki5m_ind
```

For model training run:
```
OUTPUT_DIR=/output/dir bash scripts/train_wiki.sh wiki5m_ind
```

For evaluation run:
```
bash scripts/eval.sh /path/to/model/ wiki5m_ind 
```


### Parametersettings:

Settings to run model with different context:


Applies the neighborhood sampling from the original implementation:
```
--use-neighbors
```

Adds context to head and tail sequence respectively:
```
--use-head-context
--use-tail-context
```


Limits the number of neighbors selected for context integration:
```
--max-context-size
```

This option includes the entity description for the head or tail entity (not context entities !!!)
```
--use-descriptions
```
Sets the maximum number of tokens for entity descriptions (not context descriptions !!!) (default=50)
```
--max-num-desc-tokens 50
```

This option includes the entity description for context entites:
```
--use-context-descriptions,
```

Prefix each selected neighbor with the connecting relation
```
--use-context-relation
```

Order context by relation frequency (settings are mutually exclusive)

```
--most-common-first
--least-common-first
```


## Train BERT like model from scratch
 
To initialize both enocoders with random weights add the option:

```
--custom-model-init
```


# Pretrained Models:

To select transformer model use these string as setting for "--pretrained-model":

* BERT-large: bert-large-uncased
* BERT-base: bert-base-uncased
* distilBERT: distilbert-base-uncased
* BERT-medium: prajjwal1/bert-medium
* BERT-small: prajjwal1/bert-small
* BERT-mini: prajjwal1/bert-mini
* BERT-tiny: prajjwal1/bert-tiny


The study of model size has been conducted with a fork of the official repositoy: 

The modified code can be found [here](https://github.com/till22G/SimKGC/tree/add_distilBERT_capability) in the branch: add_distilBERT_capability.

# ERNIE 2.0
To run experiments with ERNIE 2.0 use: nghuyong/ernie-2.0-base-en