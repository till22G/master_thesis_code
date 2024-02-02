import torch
import os
import json

from argparser import args
from logger import logger
from model import build_model
from help_functions import move_to_cuda


class TrainArgs():
    def __init__(self) -> None:
        pass

class EvaluationModel():

    def __init__(self) -> None:
        self.model = None
        self.train_args = TrainArgs()

    def load_checkpoint(self, checkpoint_path):
        assert os.path.exists(checkpoint_path)
        
        # load training checkpoint and restore training arguments
        checkpoint_dict = torch.load(checkpoint_path)
        self.train_args.__dict__ = checkpoint_dict["args"]
        self._set_args()

        # load model state dict
        model_state_dict = checkpoint_dict["state_dict"]
        model_state_dict = self._setup_model_state_dict(model_state_dict)

        # setup model and put it in eval mode
        self.model = build_model(self.train_args)
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.eval()

        # move to CUDA devices if available
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
            logger.info("{} cuda devices found. GPUs will be used".format(torch.cuda.device_count()))
        elif torch.cuda.device_count() == 1:
            self.model.cuda()
            logger.info("{} cuda device found. GPU will be used".format(torch.cuda.device_count()))
        else: 
            logger.info("No GPU available. CPU will be used")


    # need to check if that functions work as intended       
    def _set_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute for train arg: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))

        #self._restore_train_args()

    # need to check if that functions work as intended
    def _restore_train_args(self):
        for k, v in self.train_args.__dict__.items():
            args.__dict__[k] = v

    # if the model has been trained with nn.DataParallel keys have a "module." prefix
    def _setup_model_state_dict(self, state_dict):
        new_state_dict = {}
        # make it compatible with original version
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("hr_bert"):
                k = "bert_hr" + k[len("hr_bert"):]
            if k.startswith("tail_bert"):
                k = "bert_t" + k[len("tail_bert"):]
            new_state_dict[k] = v
        return new_state_dict

    # encode candidate entities
    @torch.no_grad()
    def encode_candidates(self, batch_candidate_datapoints) -> torch.tensor:
        encoded_entities = []
        if torch.cuda.is_available():
            batch_candidate_datapoints = move_to_cuda(batch_candidate_datapoints)
        outputs = self.model(**batch_candidate_datapoints)
        encoded_entities.append(outputs['ent_vectors'])

        return torch.cat(encoded_entities, dim=0)
    
    # encode hr_embeddings
    @torch.no_grad()
    def encode_hr(self, hr_batch) -> torch.tensor:
        hr_embeddings = []
        if torch.cuda.is_available():
            ht_batch = move_to_cuda(hr_batch)
        outputs = self.model(**hr_batch)
        hr_embeddings.append(outputs['hr_vector'])

        return torch.cat(hr_embeddings, dim=0)