from src.utils import RetrievalSystem
import json
import tqdm
import numpy as np
import pickle


qpath = "/mnt/nfs/home/randl/medrag/data/final_rephrased.json"

with open(qpath, "r") as f:
    benchmark = json.load(f)

cache = None
retriever = RetrievalSystem(retriever_name="MedCPT", corpus_name="PubMed",  db_dir="./corpus", cache = cache, HNSW=False)


embeddings_per_q = {}
for idx, question in benchmark.items():
    embeddings = []
    for rephrasing in tqdm.tqdm(question["rephrasings"]):
        print(retriever.retrievers)
        retrieved_embedding = retriever.retrievers[0][0].get_relevant_documents(rephrasing, k=16, proxcache = cache, stats = {}, tolerance = 0.0)
        embeddings.append(retrieved_embedding)
    embeddings_per_q[idx] = embeddings

with open("/mnt/nfs/home/randl/medrag/data/embeddings.json", "w") as f:
    json.dump(embeddings_per_q, f, indent=4)