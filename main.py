from src.medrag import MedRAG
import json
import tqdm
import argparse
import os
import numpy as np
import sys
import proximipy
import pickle

def load_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_dict(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

#medrag = MedRAG(llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct", rag=False)
def parse_args():
    parser = argparse.ArgumentParser(description="Construct ctx dictionary")
    parser.add_argument('--qpath', required=True)
    parser.add_argument('--trace_path', required=True)
    parser.add_argument('--acc_cache_path', required=True)
    parser.add_argument('--tolerances', default="0.0,1.0,2.0,5.0,10.0", help='Comma-separated tolerances')
    parser.add_argument('--capacity', type=int)
    parser.add_argument('--lsh_bits', type=int)
    parser.add_argument('--lsh_dim', type=int)
    parser.add_argument('--bcap', type=int)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--cache_type', choices=["LRU", "FIFO", "LSHFIFO", "LSHLRU"], required=True)
    parser.add_argument('--reranking', action='store_true', required=True)
    parser.add_argument('--dump_neighbors_path', type=str, required=False)

    parser.add_argument('--db_serve_path', required=True)
    parser.add_argument('--doc_serve_path', required=True)
    return parser.parse_args()

args = parse_args()
if (args.capacity == None):
    assert args.lsh_bits is not None
    assert args.lsh_dim is not None
    assert args.bcap is not None
    assert args.seed is not None

ctx = {
    'qpath': args.qpath,
    'trace_path': args.trace_path,
    'acc_cache_path': args.acc_cache_path,
    'tolerances': list(map(float, args.tolerances.split(','))),
    'capacity': args.capacity,
    'cache_type': args.cache_type,
    'reranking': args.reranking,
    'dump_neighbors_path' : args.dump_neighbors_path,
    'db_serve_path': args.db_serve_path,
    'doc_serve_path': args.doc_serve_path,
    'lsh_bits': args.lsh_bits,
    'lsh_dim': args.lsh_dim,
    'bcap': args.bcap,
    'seed': args.seed,
}

trace = np.load(ctx['trace_path'])

assert trace.shape[0] == 10000

if ctx['seed'] != 43: # seed 43 uses the default random trace, others shuffle it again
    np.random.seed(ctx['seed'])
    ctx['permutation'] = np.random.permutation(len(trace))
    trace = trace[ctx['permutation']]

with open(ctx['qpath'], "r") as f:
    benchmark = json.load(f)

for tolerance in ctx['tolerances']:

    medrag = MedRAG(ctx, llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct", rag=True, retriever_name="MedCPT", corpus_name="PubMed")

    if ctx['dump_neighbors_path'] is not None:
        medrag.retrieval_system.retrievers[0][0].neighbors_vectors_dump(ctx['dump_neighbors_path'])
        sys.exit()


    cache_type = ctx['cache_type']

    if cache_type == 'LRU':
        cache = proximipy.LruCache(ctx['capacity'])
    elif cache_type == 'FIFO':
        cache = proximipy.FifoCache(ctx['capacity'])
    elif cache_type == 'LSHFIFO':
        cache = proximipy.LshFifoCache(ctx['lsh_bits'], ctx['lsh_dim'], ctx['bcap'], seed = ctx['seed'])
    else:
        assert cache_type == 'LSHLRU'
        #[pyo3(signature = (num_hash, dim, bucket_capacity, seed=None))]
        cache = proximipy.LshLruCache(ctx['lsh_bits'], ctx['lsh_dim'], ctx['bcap'], seed = ctx['seed'])

    reranking = ctx['reranking']

    stats = {
        "hit_amt" : 0,
        "all_answers" : [],
        "correct_ans" : 0,
        "time_fetch_sum" : 0.0,
        "seen" : 0,
        "norm" : [],
    }

    acc_cache_path = ctx['acc_cache_path']
    acc_cache = load_dict(acc_cache_path)

    curr_seen = {}
    for amt, index in tqdm.tqdm(enumerate(list(trace))):
        if index in curr_seen:
            curr_seen[index] += 1
        else:
            curr_seen[index] = 0
        qindex = curr_seen[index]

        q = benchmark[str(index)]
        question = q['rephrasings'][qindex]
        answer = q["answer"]
        stats["seen"] += 1
        try:
            guess, _,_ = medrag.answer(question=question, k={'db': 64 if reranking else 16, 'ret':16}, proxcache = cache, tolerance = tolerance, stats = stats, acc_cache = acc_cache)
            words = guess.split()
            if len(words) == 1:
                loaded_answer = {"answer_choice" : words[0]}
            else:
                loaded_answer = json.loads(guess)
            if loaded_answer["answer_choice"] == answer:
                stats["correct_ans"] += 1
            else:
                pass
            stats['all_answers'].append({'exp' : answer, 'act' : loaded_answer['answer_choice']})

        except Exception as e:
            print(e)
            print(answer)
            if answer == "A": # guess at random, equiv to constant guess
                stats["correct_ans"] += 1
            stats['all_answers'].append({'exp' : answer, 'act' : 'A'})

        # print("hit_rate", stats["hit_amt"] / stats["seen"])
        # print("avg delay", stats["time_fetch_sum"] / stats["seen"])
        # print("accuracy", stats["correct_ans"] / stats["seen"])

        #print("norm", np.mean(stats["norm"]))

        capacity = ctx['capacity']
        seed = ctx['seed']
        if capacity is not None:
            with open(f'/mnt/nfs/home/randl/medrag/data/zipf-family{cache_type}-c{capacity}-t{tolerance}-seed{seed}.pkl', 'wb') as handle:
                pickle.dump(stats, handle)

        else:
            cbits = str(ctx['lsh_bits'])
            bcap = str(ctx['bcap'])
            named = f'bits{cbits}-cap{bcap}'
            with open(f'/mnt/nfs/home/randl/medrag/data/zipf-family{cache_type}-c{named}-t{tolerance}-seed{seed}.pkl', 'wb') as handle:
                pickle.dump(stats, handle)

        if amt % 500 == 499:
            print('saving acc')
            #save_dict(acc_cache, acc_cache_path)
            #
    print(len(cache))
