import logging
import os
import sys
import glob
import csv

import torch
import pytrec_eval
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset
from openmatch.retriever.reranker import Reranker

from openmatch.modeling import RRModelForInference

from openmatch.inference import distributed_parallel_embedding_inference

from openmatch.retriever import distributed_parallel_retrieve
from openmatch.utils import save_as_trec, load_from_trec
from copy import deepcopy
logger = logging.getLogger(__name__)


def load_beir_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as f:
        tsvreader = csv.DictReader(f, delimiter="\t")
        for row in tsvreader:
            qid = row["query-id"]
            pid = row["corpus-id"]
            rel = int(row["score"])
            if qid in qrels:
                qrels[qid][pid] = rel
            else:
                qrels[qid] = {pid: rel}
    return qrels


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    
    model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    encoding_args: EncodingArguments

    if os.path.exists(encoding_args.output_dir) and os.listdir(encoding_args.output_dir):
        if not encoding_args.overwrite_output_dir:
            logger.warning(
                f"Output directory ({encoding_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            # remove all files in the output directory
            if encoding_args.local_process_index == 0:
                for file in os.listdir(encoding_args.output_dir):
                    os.remove(os.path.join(encoding_args.output_dir, file))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    if encoding_args.fp16:
        model_args.dtype = "float16"
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("Model parameters %s", model_args)
    

    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    tokenizer.padding_side = "right"
    # tokenizer.padding_side = "left"
    split_name = "test" # typically, the evaluation split is test split, but it can be dev.  
    
    if encoding_args.phase == "rerank":
        logger.info("Reranking")
        
        qrels = {}
        qrels_path = os.path.join(data_args.data_dir, "qrels", f"{split_name}.tsv")
        
        if os.path.exists(qrels_path):
            logger.info(f"Loading {split_name} qrels")
            qrels = load_beir_qrels(qrels_path) # this is non-sharded, all process read once, cause it is small.
            qids = list(qrels.keys())
            logger.info(f"Loading {split_name} queries")
            query_dataset = InferenceDataset.load(
                tokenizer=tokenizer,
                data_args=data_args,
                data_files=os.path.join(data_args.data_dir, "queries.jsonl"),
                full_tokenization=False,
                mode="processed",
                max_len=data_args.q_max_len,
                template=data_args.query_template,
                stream=False,
                batch_size=encoding_args.per_device_eval_batch_size,
                num_processes=encoding_args.world_size,
                process_index=encoding_args.process_index,
                filter_fn=lambda x: x["_id"] in qids,
                cache_dir=data_args.data_cache_dir,
            )
            print(f"query_dataset.max_len = {query_dataset.max_len}")
        else:
            raise ValueError(f"{split_name} queries and qrels not found")
        
        logger.info("Loading corpus dataset")
        corpus_dataset = InferenceDataset.load(
            tokenizer=tokenizer,
            data_args=data_args,
            data_files=os.path.join(data_args.data_dir, "corpus.jsonl"),
            full_tokenization=False,
            mode="processed",
            max_len=data_args.p_max_len,
            template=data_args.doc_template,
            stream=False,
            batch_size=encoding_args.per_device_eval_batch_size,
            num_processes=encoding_args.world_size,
            process_index=encoding_args.process_index,
            cache_dir=data_args.data_cache_dir,
        )
        print(f"corpus_dataset.max_len = {corpus_dataset.max_len}")
        
        
        # collect trec file and compute metric for rank = 0
        # if encoding_args.process_index == 0: 
        # use glob library to to list all trec files from encoding_args.output_dir:
        partitions = glob.glob(os.path.join(encoding_args.trec_run_path, "test.*.trec"))
        run = {}
        for part in partitions:
            run.update(load_from_trec(part))
        print(os.listdir(encoding_args.trec_run_path))
    
        # sort run by score get the top encoding_args.rerank_top_k docs
        # {run: { query:  { doc:score } }
        for _query in run.keys():
            query = [(k,v)for k,v in run[_query].items()]
            query = sorted(query,key=lambda x:x[1],reverse=True)[:encoding_args.reranking_depth]
            query = {k:v for k,v in query}
            run[_query] = deepcopy(query)
        
        # num_labels = 1
        # config = AutoConfig.from_pretrained(
        #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        #     num_labels=num_labels,
        #     cache_dir=model_args.cache_dir,
        #     trust_remote_code=True
        # )
            
        model = RRModelForInference.build(
            model_args=model_args,
            cache_dir=model_args.cache_dir,
        )
        model.to(encoding_args.device)
        model.eval()
        
        reranker = Reranker(model, tokenizer, corpus_dataset, encoding_args, data_args)
        
        return_result = reranker.rerank(query_dataset, run)
        if encoding_args.trec_save_path is None:
            encoding_args.trec_save_path = os.path.join(encoding_args.output_dir, f"test.{encoding_args.process_index}.trec")
        save_as_trec(return_result, encoding_args.trec_save_path)
        
        if encoding_args.process_index == 0: 
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
            eval_results = evaluator.evaluate(return_result)

            def print_line(measure, scope, value):
                print("{:25s}{:8s}{:.4f}".format(measure, scope, value))
                # with open(
                #     os.path.join(encoding_args.output_dir, "test_result.log"), "w", encoding="utf-8"
                # ) as fw:
                #     fw.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))
            
            for query_id, query_measures in sorted(eval_results.items()):
                for measure, value in sorted(query_measures.items()):
                    pass

            for measure in sorted(query_measures.keys()):
                print_line(
                    measure,
                    "all",
                    pytrec_eval.compute_aggregated_measure(
                        measure, [query_measures[measure] for query_measures in eval_results.values()]
                    ),
                )
        
        if encoding_args.world_size > 1:
            torch.distributed.barrier()

if __name__ == "__main__":
    main()
