#!/bin/bash

# sudo apt install nano wget curl tmux git git-lfs unzip
# curl https://sh.rustup.rs -sSf | sh -s -- -y
python3 main.py \
  --qpath "/mnt/nfs/home/randl/medrag/data/full_rephrased.json" \
  --trace_path "/mnt/nfs/home/randl/medrag/data/trace.npy" \
  --acc_cache_path "/mnt/nfs/home/randl/medrag/data/acc_cache_test.json" \
  --tolerances "7.5" \
  --lsh_bits 8 \
  --lsh_dim 768 \
  --bcap 20 \
  --seed "$1" \
  --cache_type "LSHLRU" \
  --reranking \
  --db_serve_path "/mnt/nfs/home/randl/medrag/data/veconly/dbtrace.json" \
  --doc_serve_path "/mnt/nfs/home/randl/medrag/data/veconly/documents2_test.json"

