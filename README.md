## Code Repository for Master Thesis Tillman Galla

## Technical Requirements
* python>=3.7
* torch>=1.6
* transformers>=4.15

The requiremnts file for the environment used during experiments can be found here: [requirements.txt](requirements.txt)

The code has been implemented to run on CUDA devices. A implementation for CPUs has been omitted, so please make sure there are 
CUDA devices available when runnig the code.

Most experiments have been conducted using 3 RTX A6000 (48GB) GPUs.

### When encountering "CUDA out of memory" error:
    1. Make sure there are enough CUDA devices available
    2. Reduce batch size (impacts performance due to in-batch negative loss)

Following the original implementation data for WN18RR and FB15k-237 is taken from [KG-BERT](https://github.com/yao8839836/kg-bert).

Wikidata5M can be downloaded with this script taken from: https://github.com/intfloat/SimKGC/blob/main/scripts/download_wikidata5m.sh

Just run this line:

```
bash scripts/download_wikidata5m.sh
```
Tipps: 
* Run code from the master_thesis_code/rebuild_SimKGC/ directory
* Selecting datasets is case sensitive

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
bash scripts/eval.sh /path/to/model/model_checkpoint_50.mdl WN18RR 
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
bash scripts/eval.sh /path/to/model/model_checkpoint_10.mdl FB15k237 
```


### Wikidata5M-Trans

#### Preprocessing

For data preprocessing run:
```
bash scripts/preprocess.sh wiki5m_trans
```

For model training run:
Attention: be sure to specify "wiki5m_trans" to run the correct evaluation
```
OUTPUT_DIR=/output/dir bash scripts/train_wiki.sh wiki5m_trans
```

For evaluation run:
```
bash scripts/eval.sh /path/to/model/model_checkpoint_1.mdl wiki5m_trans 
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
Attention: be sure to specify "wiki5m_ind" to run the correct evaluation
```
bash scripts/eval.sh /path/to/model/model_checkpoint_1.mdl wiki5m_ind 
```


### Parametersettings:

Arguments to run model with different context:

Simply set these arguments in the respective training script to change model settings



This option includes the entity description for the head or tail entity (not context entities !!!)
Attention: The default value of this is FALSE, so if it is not set, training performance will degrade
as there are no entity descriptoins included into the input sequences if argument is ot selected.
```
--use-descriptions
```

Sets the maximum number of tokens for entity descriptions (not context descriptions !!!) (default=50)
```
--max-num-desc-tokens 50
```

Applies the neighborhood sampling from the original implementation. Can't be run together with use-head-context ot use-tail-context. 
These neighbors are not affected by other context settings. 
```
--use-neighbors
```

Adds context to head and tail sequence respectively:
```
--use-head-context
--use-tail-context
```

Limits the number of neighbors selected for context integration (affects head and tail):
```
--max-context-size
```

This option includes the entity description for context entites (affects head and tail):
```
--use-context-descriptions,
```

Prefix each selected neighbor with the connecting relation (affects head and tail):
```
--use-context-relation
```

Order context by relation frequency (settings are mutually exclusive, affects head and tail): 
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

References to thes models can be found in the thesis.

The study of model size has been conducted with a fork of the official repositoy: 

The modified code can be found [here](https://github.com/till22G/SimKGC/tree/add_distilBERT_capability) in the branch: add_distilBERT_capability.

# ERNIE 2.0
To run experiments with ERNIE 2.0 use: nghuyong/ernie-2.0-base-en


# Plotting:

The code used to create the plots in the thesis can be found [here](https://github.com/till22G/master_thesis_code/tree/main/plot_and_examination_scripts). The scripts can be run with:

```
python3 script.py --task task
```

The individual settings for each plotting script can be found at the top of the respective python file.
