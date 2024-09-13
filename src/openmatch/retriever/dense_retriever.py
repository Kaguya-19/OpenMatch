import gc
import glob
import logging
import os
import pickle
from contextlib import nullcontext
from typing import Dict, List, Union

import faiss
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..arguments import InferenceArguments as EncodingArguments
from ..dataset import DRInferenceCollator
from ..modeling import DRModelForInference, DROutput
from ..utils import merge_retrieval_results_by_score

logger = logging.getLogger(__name__)

def _retrieve_one_shard(
    corpus_shard_path: str,
    encoded_queries_tensor: torch.Tensor,
    topk: int,
    device: str,
):
    doc_lookup = []
    with open(corpus_shard_path, "rb") as f:
        data = pickle.load(f)
    encoded_corpus = data[0]
    corpus_lookup_indices = data[1]
    dim = encoded_corpus.shape[1] # dimension of encodings
    
    # now I have a numpy array, I need to convert it to torch tensor, and then to cuda
    encoded_corpus_tensor = torch.tensor(encoded_corpus, device=device)
    
    # compute the inner product of the query and corpus
    scores = torch.matmul(encoded_queries_tensor, encoded_corpus_tensor.T)
    # get the topk scores and indices
    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)
    del encoded_corpus, encoded_corpus_tensor, scores
    gc.collect()
    return topk_scores.clone(), topk_indices.clone(), corpus_lookup_indices # return the cloned tensor, then the inner tensor will be destroyed


def distributed_parallel_retrieve(
    args: EncodingArguments,
    topk: int,
):
    with torch.no_grad():
        final_result = {}
        
        # step1: this process only load its own sharded queriess
        encoded_queries = [] # this is persistent
        query_lookup = [] # this is persistent as well
        
        # - use glob to list all the partitions like embeddings.query.{process_index}: (belong to this process)
        query_all_partitions = glob.glob(
            os.path.join(args.output_dir, f"embeddings.query.rank.{args.process_index}*")
        )
        
        logger.info(f"query_all_partitions = {query_all_partitions}")
        if len(query_all_partitions) == 0:
            return {}
        for part in query_all_partitions:
            with open(part, "rb") as f:
                data = pickle.load(f)
            query_lookup_indices = data[1]
            if len(query_lookup_indices) == 0:  # No data
                continue
            encoded_queries.append(data[0])
            query_lookup.extend(query_lookup_indices)
        encoded_queries_all = np.concatenate(encoded_queries) # this is persistent
        # - now convert it to torch tensor, and then to cuda
        encoded_queries_tensor = torch.tensor(encoded_queries_all, device=args.device) # this is persistent
        
        # step2: iterate corpus partitions
        corpus_all_partitions = glob.glob(
            os.path.join(args.output_dir, "embeddings.corpus.rank.*")
        )
        if len(corpus_all_partitions) == 0:
            raise ValueError("No pre-computed document embeddings found")

        logger.info(f"corpus_all_partitions = {corpus_all_partitions}")
        
        # a dict to store the final result, namely, cur_result
        cur_result = {}
        for qid in query_lookup:
            cur_result[qid] = {}
        
        for i, part in enumerate(corpus_all_partitions):
            topk_scores, topk_indices, corpus_lookup_indices = _retrieve_one_shard(
                corpus_shard_path=part, 
                encoded_queries_tensor=encoded_queries_tensor, 
                topk=topk,
                device=args.device
            )
            
            # - update the cur_result by topk_scores and topk_indices:
            for q in range(topk_scores.shape[0]):
                qid = query_lookup[q]
                for idx, score in zip(topk_indices[q], topk_scores[q]):
                    idx = corpus_lookup_indices[idx.item()]
                    cur_result[qid][idx] = score.item()
            
            del topk_scores, topk_indices, corpus_lookup_indices # release the tensor
            gc.collect()
            
            # then I only need the topk for each qid:
        return cur_result

class Retriever:
    def __init__(
        self, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments
    ):
        logger.info("Initializing retriever")
        self.model = model
        self.corpus_dataset = corpus_dataset
        self.args = args
        self.doc_lookup = []
        self.query_lookup = []
        self.index_on_gpu = False

        self.model.to(self.args.device)
        self.model.eval()

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        logger.info("Moving index to GPU(s)")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        if self.args.faiss_starting_gpu >= ngpu:
            raise ValueError(
                f"Faiss GPU {self.args.faiss_starting_gpu} is out of range (0-{ngpu-1})"
            )
        for i in range(self.args.faiss_starting_gpu, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)
        self.index_on_gpu = True

    def doc_embedding_inference(self):
        # Note: during evaluation, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if self.corpus_dataset is None:
            raise ValueError("No corpus dataset provided")
        dataloader = DataLoader(
            self.corpus_dataset,
            # Note that we do not support DataParallel here
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=DRInferenceCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)
        encoded = []
        lookup_indices = []
        idx = 0
        prev_idx = 0
        for batch_ids, batch in tqdm(dataloader, disable=self.args.process_index > 0):
            lookup_indices.extend(batch_ids)
            idx += len(batch_ids)
            with amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    model_output: DROutput = self.model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())
            if len(lookup_indices) >= self.args.max_inmem_docs // self.args.world_size:
                encoded = np.concatenate(encoded)
                with open(
                    os.path.join(
                        self.args.output_dir,
                        "embeddings.corpus.rank.{}.{}-{}".format(
                            self.args.process_index, prev_idx, idx
                        ),
                    ),
                    "wb",
                ) as f:
                    pickle.dump((encoded, lookup_indices), f, protocol=4)
                encoded = []
                lookup_indices = []
                prev_idx = idx
                gc.collect()

        if len(lookup_indices) > 0:
            encoded = np.concatenate(encoded)
            with open(
                os.path.join(
                    self.args.output_dir,
                    "embeddings.corpus.rank.{}.{}-{}".format(
                        self.args.process_index, prev_idx, idx
                    ),
                ),
                "wb",
            ) as f:
                pickle.dump((encoded, lookup_indices), f, protocol=4)

        del encoded
        del lookup_indices

        if self.args.world_size > 1:
            torch.distributed.barrier()

    def init_index_and_add(self, partition: str = None):
        logger.info("Initializing Faiss index from pre-computed document embeddings")
        partitions = (
            [partition]
            if partition is not None
            else glob.glob(os.path.join(self.args.output_dir, "embeddings.corpus.rank.*"))
        )
        for i, part in enumerate(partitions):
            with open(part, "rb") as f:
                data = pickle.load(f)
            encoded = data[0]
            lookup_indices = data[1]
            if i == 0:
                dim = encoded.shape[1]
                self._initialize_faiss_index(dim)
            self.index.add(encoded)
            self.doc_lookup.extend(lookup_indices)

    @classmethod
    def build_all(
        cls, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments
    ):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        if args.process_index == 0:
            retriever.init_index_and_add()
            if retriever.args.use_gpu and not retriever.index_on_gpu:
                retriever._move_index_to_gpu()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    @classmethod
    def build_embeddings(
        cls, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments
    ):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model: DRModelForInference, args: EncodingArguments):
        retriever = cls(model, None, args)
        if args.process_index == 0:
            retriever.init_index_and_add()
            if retriever.args.use_gpu and not retriever.index_on_gpu:
                retriever._move_index_to_gpu()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    def reset_index(self):
        if self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []
        self.index_on_gpu = False

    def query_embedding_inference(self, query_dataset: IterableDataset):
        dataloader = DataLoader(
            query_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=DRInferenceCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        encoded = []
        lookup_indices = []
        for batch_ids, batch in tqdm(dataloader, disable=self.args.process_index > 0):
            lookup_indices.extend(batch_ids)
            with amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    if not self.args.encode_query_as_passage:
                        model_output: DROutput = self.model(query=batch)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output: DROutput = self.model(passage=batch)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())

        if len(encoded) > 0:  # If there is no data in the process, we don't do anything
            encoded = np.concatenate(encoded)

        with open(
            os.path.join(
                self.args.output_dir, "embeddings.query.rank.{}".format(self.args.process_index)
            ),
            "wb",
        ) as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)

        if self.args.world_size > 1:
            torch.distributed.barrier()

    def search(self, topk: int = 100):
        logger.info("Searching")
        if self.index is None:
            raise ValueError("Index is not initialized")
        encoded = []
        for i in range(self.args.world_size):
            with open(
                os.path.join(self.args.output_dir, "embeddings.query.rank.{}".format(i)), "rb"
            ) as f:
                data = pickle.load(f)
            lookup_indices = data[1]
            if len(lookup_indices) == 0:  # No data
                continue
            encoded.append(data[0])
            self.query_lookup.extend(lookup_indices)
        encoded = np.concatenate(encoded)

        return_dict = {}
        D, I = self.index.search(encoded, topk)
        original_indices = np.array(self.doc_lookup)[I]
        q = 0
        for scores_per_q, doc_indices_per_q in zip(D, original_indices):
            qid = str(self.query_lookup[q])
            return_dict[qid] = {}
            for doc_index, score in zip(doc_indices_per_q, scores_per_q):
                doc_index = str(doc_index)
                if self.args.remove_identical and qid == doc_index:
                    continue
                return_dict[qid][doc_index] = {"score": float(score)}
            q += 1

        logger.info("End searching with {} queries".format(len(return_dict)))

        return return_dict

    @staticmethod
    def fill_retrieval_result_with_document_texts(
        retrieval_result: Union[
            Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]
        ],
        doc_id_to_doc,
        single_query: bool = False,
    ):
        if single_query:
            for doc_id, score in retrieval_result.items():
                retrieval_result[doc_id] = (
                    {"score": score["score"], **doc_id_to_doc[doc_id]}
                    if isinstance(doc_id_to_doc[doc_id], Dict)
                    else {"score": score["score"], "text": doc_id_to_doc[doc_id]}
                )
        else:
            for query_id, doc_id_to_score in retrieval_result.items():
                for doc_id, score in doc_id_to_score.items():
                    retrieval_result[query_id][doc_id] = (
                        {"score": score["score"], **doc_id_to_doc[doc_id]}
                        if isinstance(doc_id_to_doc[doc_id], Dict)
                        else {"score": score["score"], "text": doc_id_to_doc[doc_id]}
                    )

    def retrieve(
        self,
        query_dataset: IterableDataset = None,
        query: str = None,
        tokenizer: PreTrainedTokenizer = None,
        doc_id_to_doc=None,
        topk: int = 100,
    ):
        if query_dataset is not None:
            self.query_embedding_inference(query_dataset)
            result = {}
            if self.args.process_index == 0:
                result = self.search(topk)
                if doc_id_to_doc is not None:
                    self.fill_retrieval_result_with_document_texts(result, doc_id_to_doc)
            if self.args.world_size > 1:
                torch.distributed.barrier()
            return result

        query_tokenized = tokenizer(query, return_tensors="pt")
        for k, v in query_tokenized.items():
            query_tokenized[k] = v.to(self.args.device)
        model_output: DROutput = self.model(query=query_tokenized)
        D, I = self.index.search(model_output.q_reps.cpu().detach().numpy(), topk)
        original_indices = np.array(self.doc_lookup)[I]
        D, original_indices = D[0].tolist(), original_indices[0].tolist()
        result = {}
        for score, index in zip(D, original_indices):
            result[index] = {"score": score}
        if doc_id_to_doc is not None:
            self.fill_retrieval_result_with_document_texts(result, doc_id_to_doc, single_query=True)
        return result

    def split_retrieve(self, query_dataset: IterableDataset, topk: int = 100):
        self.query_embedding_inference(query_dataset)
        final_result = {}
        if self.args.process_index == 0:
            all_partitions = glob.glob(
                os.path.join(self.args.output_dir, "embeddings.corpus.rank.*")
            )
            for partition in all_partitions:
                logger.info("Loading partition {}".format(partition))
                self.init_index_and_add(partition)
                if self.args.use_gpu:
                    self._move_index_to_gpu()
                cur_result = self.search(topk)
                self.reset_index()
                final_result = merge_retrieval_results_by_score([final_result, cur_result], topk)
        if self.args.world_size > 1:
            torch.distributed.barrier()
        return final_result


class SuccessiveRetriever(Retriever):
    def __init__(
        self, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments
    ):
        super().__init__(model, corpus_dataset, args)

    @classmethod
    def build_all(
        cls, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments
    ):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model: DRModelForInference, args: EncodingArguments):
        retriever = cls(model, None, args)
        return retriever

    def retrieve(self, query_dataset: IterableDataset, topk: int = 100):
        self.query_embedding_inference(query_dataset)
        final_result = {}
        if self.args.process_index == 0:
            all_partitions = glob.glob(
                os.path.join(self.args.output_dir, "embeddings.corpus.rank.*")
            )
            for partition in all_partitions:
                logger.info("Loading partition {}".format(partition))
                self.init_index_and_add(partition)
                if self.args.use_gpu:
                    self._move_index_to_gpu()
                cur_result = self.search(topk)
                self.reset_index()
                final_result = merge_retrieval_results_by_score([final_result, cur_result], topk)
        if self.args.world_size > 1:
            torch.distributed.barrier()
        return final_result
