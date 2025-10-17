from src.medrag import MedRAG
import json
import tqdm
import os
import numpy as np
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

medrag = MedRAG(llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct", rag=True, retriever_name="MedCPT", corpus_name="PubMed")
#medrag = MedRAG(llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct", rag=False)

qpath = "/mnt/nfs/home/randl/medrag/data/full_rephrased.json"
trace = "/mnt/nfs/home/randl/medrag/data/trace.npy"
trace = np.load(trace)

assert trace.shape[0] == 10000

with open(qpath, "r") as f:
    benchmark = json.load(f)

tolerances = [0.0]
capacity = 200
curr_seen = {}



for tolerance in tolerances:

    cache = proximipy.LRUCache(capacity)
    reranking = True

    stats = {
        "hit_amt" : 0,
        "correct_ans" : 0,
        "time_fetch_sum" : 0.0,
        "seen" : 0,
        "norm" : [],
        # "vecs" : [],
        # "scores" : [],
    }
    # for i, q in benchmark.items():
    # for question in tqdm.tqdm(q["rephrasings"]):
    

    for index in list(trace):
        if index in curr_seen:
            curr_seen[index] += 1
        else:
            curr_seen[index] = 0
        qindex = curr_seen[index]

        q = benchmark[str(index)]
        question = q['rephrasings'][qindex]
        answer = q["answer"]
        stats["seen"] += 1
            guess, _,_ = medrag.answer(question=question, k={'db': 64 if reranking else 16, 'ret':16}, proxcache = cache, tolerance = tolerance, stats = stats, acc_cache = acc_cache)
       
    

        with open(f'/mnt/nfs/home/randl/medrag/data/veconly/dbtrace.json', 'w') as handle:
            json.dump({
                'vecs' : np.array(retrieved).tolist(),
                'scores' : np.array(scores).tolist(),
                'timings' : medrag.retrieval_system.retrievers[0][0].timing
            }, handle)
            
