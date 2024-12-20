# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import argparse
import csv
import logging
import pickle

import numpy as np
import torch
from tqdm import tqdm
from ipdb import set_trace as bp
import transformers

import index_utils.slurm
import index_utils.contriever
import index_utils.utils
import index_utils.data_contriever
import index_utils.normalize_text

def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in tqdm(enumerate(passages)):
            batch_ids.append(p['id'])
            if args.no_title or not 'title' in p:
                text = p['text']
            else:
                text = p['title'] + ' ' + p['text']
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = index_utils.normalize_text.normalize(text)
            batch_text.append(text)

            # Process the batch when the batch size is reached or it's the last item
            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                # Send tensors to GPU, apply half() only to floating-point tensors
                for key in encoded_batch:
                    encoded_batch[key] = encoded_batch[key].cuda()
                    if encoded_batch[key].dtype == torch.float32 and not args.no_fp16:
                        encoded_batch[key] = encoded_batch[key].half()  # Convert to float16 only if it's a float tensor

                try:
                    # Compute embeddings
                    model_output = model(**encoded_batch)
                    # CLS pooling (use only CLS token embeddings)
                    embeddings = model_output[0][:, 0]

                    total += len(batch_ids)
                    allids.extend(batch_ids)
                    allembeddings.append(embeddings.cpu())  # Move to CPU to save GPU memory

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"Out of memory at batch {k}. Reducing batch size and retrying.")
                        args.per_gpu_batch_size = max(1, args.per_gpu_batch_size // 2)  # Reduce batch size
                        batch_ids = []  # Reset batch
                        batch_text = []  # Reset batch
                        torch.cuda.empty_cache()
                        continue  # Skip to the next iteration after clearing memory
                    else:
                        raise e  # Reraise if it's a different error

                # Reset for the next batch
                batch_text = []
                batch_ids = []

                if k % 100000 == 0 and k > 0:
                    print(f'Encoded passages {total}', flush=True)

    # Concatenate embeddings and convert to numpy array
    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings



def main(args):
    model, tokenizer = index_utils.contriever.load_retriever(args.model_name_or_path)
    print(f'Model loaded from {args.model_name_or_path}.', flush=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    if args.passages == "wikitext":
        passages = index_utils.data_contriever.process_huggingface_dataset(args.passages, args.chunk_size)
    else:
        passages = index_utils.data_contriever.load_passages(args.passages)

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards-1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f'Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.')

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    save_file = os.path.join(args.output_dir, args.prefix + f'_{args.shard_id:02d}')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Saving {len(allids)} passage embeddings to {save_file}.')
    with open(save_file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f)

    print(f'Total passages processed {len(allids)}. Written to {save_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.tsv file)')
    parser.add_argument('--output_dir', type=str, default='wikipedia_embeddings', help='dir path to save embeddings')
    parser.add_argument('--prefix', type=str, default='passages', help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=512, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--passage_maxlength', type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument('--chunk_size', type=int, default=64, help="Maximum number of words in a chunk")
    parser.add_argument('--model_name_or_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--no_title', action='store_true', help="title not added to the passage body")
    parser.add_argument('--lowercase', action='store_true', help="lowercase text before encoding")
    parser.add_argument('--normalize_text', action='store_true', help="lowercase text before encoding")

    args = parser.parse_args()

    index_utils.slurm.init_distributed_mode(args)

    main(args)
