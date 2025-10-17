import argparse
import numpy as np
import diskannpy
import proximipy
import inspect

def get_parser():
    parser = argparse.ArgumentParser(description="Load a disk index and process vectors with caching.")
    parser.add_argument("--cache_size", type=int, default=250, help="Size of the cache.")
    parser.add_argument("--cache_tolerance", type=float, default=5.0, help="Tolerance level for caching.")
    parser.add_argument("--vectors_path", type=str, required=True, help="Path to the input vectors file (NumPy .npy format).")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the disk index directory (there should be several files).")
    parser.add_argument("--search_threads", type=int, required=True, help="Amount of threads used for seraching in the DB")
    parser.add_argument("--distance_metric", choices=["l2", "mips", "cosine"], default="l2", help="Path to the input vectors file (NumPy .npy format).")
    parser.add_argument("--db_return_k", type=int, default=16, help="Amount of vectors to return after search")
    parser.add_argument("--db_search_k", type=int, default=32, help="Amount of vectors to consider when searching. Must be >= db_return_k")

    return parser

def k_recall(relevant_items, retrieved_items):
    k = len(relevant_items) 
    if k == 0:
        return 0.0

    retrieved_set = set(retrieved_items)
    relevant_set = set(relevant_items)
    intersection = len(retrieved_set & relevant_set)    
    return intersection / k

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    assert args.cache_size > 0, "Cache size must be greater than 0."
    assert args.cache_tolerance > 0.0, "Cache tolerance must be greater than 0."
    
    cache = proximipy.LshLruCache(num_hash=8, dim=768, bucket_capacity=20, seed = 42)  
    
    input_vectors = np.load(args.vectors_path)
    print(input_vectors.shape)

    index = diskannpy.StaticDiskIndex(args.index_path, args.search_threads, 256, distance_metric=args.distance_metric, vector_dtype=np.float32)
    
    results = {
        "hit" : 0,
        "recall" : []
    }

    for ite, v in enumerate(input_vectors):
        # complexity = the amount of vectors the db considers when searching. 
        # reranked to k_neihgbors just before returning
        db_neighs = index.search(query = v, k_neighbors=args.db_return_k, complexity=args.db_search_k).identifiers

        cache_neighs = cache.find(list(v))
        if cache_neighs is None:
            cache.insert(list(v), db_neighs, args.cache_tolerance)
        else:
            results["hit"] += 1
            results["recall"].append(k_recall(db_neighs, cache_neighs))
            
        if ite % 5000 == 4999:
            print(ite, results["hit"], np.mean(results["recall"]))

    print('final results :', results["hit"], np.mean(results["recall"]))
        

if __name__ == "__main__":
    main()
