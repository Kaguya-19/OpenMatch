import copy
import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5EncoderModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments, ModelArguments
from ..arguments import RRTrainingArguments as TrainingArguments
from ..loss import CrossEntropyLoss, rr_loss_functions
from ..utils import mean_pooling
from .linear import LinearHead

logger = logging.getLogger(__name__)


@dataclass
class RROutput(ModelOutput):
    pos_pair_scores: Tensor = None
    neg_pair_scores: Tensor = None
    loss: Tensor = None


class RRModel(nn.Module):
    def __init__(
        self,
        lm_r: PreTrainedModel,
        head: nn.Module = None,
        feature: str = "last_hidden_state",
        pooling: str = "first",
        attention: str = "bidirectional",
        pos_token: str = None,
        neg_token: str = None,
        tokenizer: PreTrainedTokenizer = None,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_r = lm_r
        self.head = head

        self.feature = feature
        self.pooling = pooling
        self.attention = attention

        self.pos_token = pos_token
        self.neg_token = neg_token
        self.tokenizer = tokenizer
        self.pos_token_id = (
            tokenizer.encode(self.pos_token, add_special_tokens=False)[0]
            if self.pos_token
            else None
        )
        self.neg_token_id = (
            tokenizer.encode(self.neg_token, add_special_tokens=False)[0]
            if self.neg_token
            else None
        )

        self.model_args = model_args
        self.data_args = data_args
        self.train_args = train_args

        if train_args is not None:
            self.loss_fn_str = train_args.loss_fn
            self.loss_fn = rr_loss_functions[self.loss_fn_str]()
            self.margin = train_args.margin

        if "T5" in type(self.lm_r).__name__ and not self.model_args.encoder_only:
            self.loss_fn_str = "ce"
            self.loss_fn = CrossEntropyLoss()

    def _get_config_dict(self):
        config = {
            "plm_backbone": {
                "type": type(self.lm_r).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "pos_token": self.pos_token,
            "neg_token": self.neg_token,
        }
        return config

    def forward(
        self,
        pos_pairs: Dict[str, Tensor] = None,
        neg_pairs: Dict[str, Tensor] = None,
    ):
        pos_pair_scores = self.encode(pos_pairs)
        neg_pair_scores = self.encode(neg_pairs)

        if self.loss_fn_str in ["mr", "smr"]:
            loss = self.loss_fn(pos_pair_scores, neg_pair_scores, margin=self.margin)
        else:
            loss = self.loss_fn(pos_pair_scores, neg_pair_scores)
        
        with torch.no_grad():
            pos_per_neg = neg_pair_scores.shape[0] // pos_pair_scores.shape[0]

        return RROutput(
            loss=loss,
            pos_pair_scores=pos_pair_scores,
            neg_pair_scores=neg_pair_scores,
        )

    def encode(self, items):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        if "T5" in type(self.lm_r).__name__ and not self.model_args.encoder_only:
            decoder_input_ids = torch.zeros((items.input_ids.shape[0], 1), dtype=torch.long).to(
                items.input_ids.device
            )
            items_out = self.lm_r(**items, decoder_input_ids=decoder_input_ids, return_dict=True)
            logits = items_out.logits
            scores = logits[:, 0, [self.neg_token_id, self.pos_token_id]]  # batch_size * 2
        else:
            items_out = self.lm_r(**items, return_dict=True)
            hidden = getattr(items_out, self.feature)
            if self.pooling == "first":
                reps = hidden[:, 0, :]
            elif self.pooling == "mean":
                reps = mean_pooling(hidden, items.attention_mask)
            elif self.pooling == "no":
                reps = hidden
            else:
                raise ValueError("Unknown pooling type: {}".format(self.pooling))
            scores = self.head(reps) if self.head is not None else reps  # batch_size * 1
        return scores

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        model_name_or_path: str = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        tokenizer: PreTrainedTokenizer = None,
        **hf_kwargs,
    ):
        # load local
        config = None
        model_class = None

        if os.path.exists(os.path.join(model_args.model_name_or_path, "openmatch_config.json")):
            with open(os.path.join(model_args.model_name_or_path, "openmatch_config.json")) as f:
                config = json.load(f)

        # an OpenMatch model
        if model_args.attention == "bidirectional":
            config.is_causal = False
        elif model_args.attention == "causal":
            # config.is_causal = True
            pass
        else:
            raise ValueError(f"attention type {model_args.attention} is not valid")
        
        if os.path.isdir(model_args.model_name_or_path) and config is not None:
            logger.info(f"loading reranking model weight from {model_args.model_name_or_path}")
            model_name = config["plm_backbone"]["type"]
            model_class = getattr(importlib.import_module("transformers"), model_name)
            if model_args.dtype == "float16":
                lm_r = model_class.from_pretrained(
                model_args.model_name_or_path, 
                trust_remote_code=True,
                attn_implementation=model_args.attn_implementation, 
                config=config,
                torch_dtype=torch.float16,
                **hf_kwargs
            )
            elif model_args.dtype == 'bfloat16':
                lm_r = model_class.from_pretrained(
                model_args.model_name_or_path, 
                trust_remote_code=True,
                attn_implementation=model_args.attn_implementation, 
                config=config,
                torch_dtype=torch.bfloat16,
                **hf_kwargs
                )
            else:
                lm_r = model_class.from_pretrained(
                model_args.model_name_or_path, 
                trust_remote_code=True,
                attn_implementation=model_args.attn_implementation, 
                config=config,
                **hf_kwargs
            )
            head = (
                LinearHead.load(ckpt_dir=model_args.model_name_or_path)
                if os.path.exists(os.path.join(model_args.model_name_or_path, "head_config.json"))
                else None
            )
        else:  # a Huggingface model
            hf_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            if model_args.encoder_only:
                model_class = T5EncoderModel
                head = LinearHead(model_args.projection_in_dim, 1)
            elif (
                hf_config.architectures is not None and "T5" in hf_config.architectures[0]
            ):  # Pre-trained T5 model
                model_class = T5ForConditionalGeneration
                head = None
            else:
                model_class = AutoModel
                head = LinearHead(model_args.projection_in_dim, 1)
            lm_r = model_class.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        model = cls(
            lm_r=lm_r,
            head=head,
            feature=model_args.feature if config is None else config["plm_backbone"]["feature"],
            pooling=model_args.pooling if config is None else config["pooling"],
            attention=model_args.attention,
            pos_token=model_args.pos_token if config is None else config["pos_token"],
            neg_token=model_args.neg_token if config is None else config["neg_token"],
            tokenizer=tokenizer,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        self.lm_r.save_pretrained(output_dir)
        if self.head is not None:
            self.head.save(output_dir)

        with open(os.path.join(output_dir, "openmatch_config.json"), "w") as f:
            json.dump(self._get_config_dict(), f, indent=4)
            
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        gradient_checkpointing_kwargs["use_reentrant"] = False # handle a bug with DDP
        self.lm_r.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs) # this model should be transformers model
        return
    
class RRModelForInference(RRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head.eval()
        self.head = self.head.half()
        self.head.eval()
        self.eval()

    @torch.no_grad()
    def encode(self, items):
        return super(RRModelForInference, self).encode(items)

    def forward(
        self,
        pos_pairs: Dict[str, Tensor] = None,
        neg_pairs: Dict[str, Tensor] = None,
    ):
        pos_pair_scores = self.encode(pos_pairs)
        neg_pair_scores = self.encode(neg_pairs)
        return RROutput(
            pos_pair_scores=pos_pair_scores,
            neg_pair_scores=neg_pair_scores,
        )