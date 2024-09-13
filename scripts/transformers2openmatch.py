import jsonlines
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import numpy as np
import logging
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import json
from fire import Fire

logger = logging.getLogger(__name__)

class LinearHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
    ):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.config = {"input_dim": input_dim, "output_dim": output_dim}

    def forward(self, rep: Tensor = None):
        return self.linear(rep)

    @classmethod
    def load(cls, ckpt_dir: str):
        logger.info(f"Loading linear head from {ckpt_dir}")
        model_path = os.path.join(ckpt_dir, "linear.pt")
        config_path = os.path.join(ckpt_dir, "head_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(model_path))
        return model

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, "linear.pt"))
        with open(os.path.join(save_path, "head_config.json"), "w") as f:
            json.dump(self.config, f, indent=4)


def main(transformers_path:str, openmatch_path:str, input_dim:int=768):
    sequenceclassification_model =  AutoModelForSequenceClassification.from_pretrained(transformers_path,trust_remote_code=True, torch_dtype=torch.bfloat16)
    head = LinearHead(input_dim=input_dim, output_dim=1)
    head.linear = sequenceclassification_model.score
    
    head.save(openmatch_path)
    sequenceclassification_model.model.save_pretrained(openmatch_path)

if __name__ == "__main__":
    Fire(main)