#!/bin/bash

# sudo apt install nano wget curl tmux git git-lfs unzip
# curl https://sh.rustup.rs -sSf | sh -s -- -y



python3 main.py \
  --qpath "/mnt/nfs/home/randl/medrag/data/full_rephrased.json" \
  --trace_path "/mnt/nfs/home/randl/medrag/data/trace.npy" \
  --acc_cache_path "/mnt/nfs/home/randl/medrag/data/acc_cache.json" \
  --tolerances "0.0,1.0,2.0,5.0,10.0" \
  --capacity 200 \
  --cache_type "LRU" \
  --seed "$1" \
  --reranking
