## Code Repository for Master Thesis Tillman Galla

First draft of README.md for code repositiry


## Technical Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

The code has been implemented to run with CUDA devices. A implementation for CPUs has been omitted, so please make sure there are 
CUDA devices available when runnig the code.

Most experiments have been conducted using 3 RTX A6000 (48GB) GPUs

When encountering "CUDA out of memory" error:
    1. Make sure there are enough CUDA devices available
    2. Reduce batch size (impacts performance due to in-batch negative loss)

## Preprocessing

Following the original implementation data for WN18RR and FB15k-237 is taken from [KG-BERT](https://github.com/yao8839836/kg-bert).

### WN18RR

```
bash scripts/preprocessing.sh WN18RR
```

```
bash scripts/preprocessing.sh FB15k237
```

```
bash scripts/preprocessing.sh wiki5m_ind
```

```
bash scripts/preprocessing.sh wiki5m_trans
```