import os
import faiss
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from model import encode_sentences, load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple, Optional

np.random.seed(0)


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        dataset: List[Dict] = [json.loads(l) for l in lines]
    return dataset

def tsne_reduce_dataset(embeddings: np.ndarray, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
    embeddings = PCA(n_components=50).fit_transform(embeddings)
    reduction: TSNE = TSNE(n_components=n_components, perplexity=perplexity)
    disp_embeds: np.ndarray = reduction.fit_transform(embeddings)
    return disp_embeds

def save_embeds(sentences: List[str], embeddings: np.ndarray, categories: List[str], outfile: str = "embeddings.pkl"):
    with open(outfile, "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'categories': categories}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def load_embeds(embed_file: str = "embeddings.pkl") -> Tuple[List[str], np.ndarray]:
    with open(embed_file, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data['categories'], stored_data['sentences'], stored_data['embeddings']

def balance_dataset_idx(categories: List[str], num_sentences: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random
    unique_categories = list(set(categories))
    num_per_cat = num_sentences // len(unique_categories)

    np_cat = np.array(categories)
    idxs = [(np_cat == cat).nonzero()[0] for cat in unique_categories]
    sel_idx = [idx[rng.choice(len(idx), size=min(num_per_cat, len(idx)), replace=False)] for idx in idxs]
    return np.concatenate(sel_idx)

def get_category_idxs(categories: np.ndarray) -> Dict:
    unique_categories = list(set(categories))
    return {cat: (categories == cat).nonzero() for cat in unique_categories}

def build_faiss_index(index_str: str, embeddings: np.ndarray):
    index = faiss.index_factory(embeddings.shape[1], index_str)
    index.add_with_ids(embeddings, np.arange(embeddings.shape[0]))
    return index

if __name__ == "__main__":
    embed_file = input("Input Embedding Filename: ")
    dataset_path = input("Input Dataset Path: ")
    dataset: List[Dict] = load_dataset(dataset_path)
    model: SentenceTransformer = load_model()

    num_sentences: int = 1000

    if os.path.isfile(embed_file):
        print("Found Existing Embeddings")
        categories, dataset_sentences, embeddings = load_embeds(embed_file)
    else:
        # dataset_sentences: np.ndarray = np.array([f"{data['headline']} -- {data['short_description']}" for data in dataset])
        dataset_sentences: np.ndarray = np.array([data['headline'] for data in dataset])
        categories: np.ndarray = np.array([data['category'] for data in dataset])
        urls: np.ndarray = np.array([data['link'] for data in dataset])
        
        empty_sentences = (dataset_sentences == '')
        dataset_sentences = dataset_sentences[~empty_sentences]
        categories = categories[~empty_sentences]

        # idx = balance_dataset_idx(categories, num_sentences)
        # dataset_sentences = dataset_sentences[idx]
        # categories = categories[idx]
        # urls = urls[idx]

        print("Generating Text Embeddings")
        embeddings = encode_sentences(model, dataset_sentences.tolist())
        print("Saving Embeddings")
        save_embeds(dataset_sentences, embeddings, categories, embed_file)

        query_data = {
            "embed_file": embed_file,
            "sentences": dataset_sentences.tolist(), 
            "categories": categories.tolist(), 
            "urls": urls.tolist(), 
            "misinf_ratio": np.zeros(len(urls)).tolist(), 
            "n_queries": np.zeros(len(urls)).tolist()
        }
        with open(f"search_full.json", "w") as f:
            f.write(json.dumps(query_data))

    idx = balance_dataset_idx(categories, num_sentences)
    cat_sample = categories[idx]
    embed_sample = embeddings[idx]
    sentence_sample = dataset_sentences[idx]

    print("Reducing Embeddings to Display")
    disp_embeddings = tsne_reduce_dataset(embed_sample, n_components=3)
    category_idx = get_category_idxs(cat_sample)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for category, idx in category_idx.items():
        cat_data = disp_embeddings[idx]
        x, y, z = cat_data.T
        ax.scatter(x, y, z, label=category)
    plt.legend()
    plt.show()

    index = build_faiss_index("HNSW64,IDMap", embeddings)
    faiss.write_index(index, f"search_full.index")
    query_str = input("Input Query String: ")
    query_embed = encode_sentences(model, [query_str])
    dist, idx = index.search(query_embed, k=5)
    for i in idx:
        print(dataset_sentences[i])