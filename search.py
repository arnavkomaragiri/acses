import json
import glob
import faiss
import argparse

import numpy as np
import matplotlib.pyplot as plt

from model import encode_sentences, load_model
from store_visualize import load_embeds, tsne_reduce_dataset
from typing import List, Tuple, Dict

def load_index(index_path: str):
    return faiss.read_index(index_path)

def load_query_data(data_path: str):
    with open(data_path, "r") as f:
        data = json.load(f)
    data['sentences'] = np.array(data['sentences'])
    data['categories'] = np.array(data['categories'])
    data['urls'] = np.array(data['urls'])
    data['misinf_ratio'] = np.array(data['misinf_ratio'])
    data['n_queries'] = np.array(data['n_queries'])
    return data

def save_query_data(data_path: str, query_data: Dict):
    data['sentences'] = data['sentences'].tolist()
    data['categories'] = data['categories'].tolist()
    data['urls'] = data['urls'].tolist()
    data['misinf_ratio'] = data['misinf_ratio'].tolist()
    data['n_queries'] = data['n_queries'].tolist()
    with open(data_path, "w") as f:
        f.write(json.dumps(query_data))


def query_index(index, query: str, query_data: Dict, k: int = 5) -> Tuple:
    embedding = encode_sentences(model, [query])
    scores, idx = index.search(embedding, k=k)
    misinf_ratios = query_data['misinf_ratio'][idx]
    return query_data['sentences'][idx], query_data['categories'][idx], query_data['urls'][idx], scores, misinf_ratios > 0.8, idx

def get_index_query_files(pattern: str = "search_*"):
    index_pattern, data_pattern = pattern + ".index", pattern + ".json"
    index_files, data_files = glob.glob(index_pattern), glob.glob(data_pattern)

    get_version = lambda s: int(s.split('_')[-1].split('.')[0])
    index_versions, data_versions = [get_version(s) for s in index_files], [get_version(s) for s in data_files]

    index_versions = [i for i in index_versions if i in data_versions]
    data_versions = [d for d in data_versions if d in index_versions]

    max_version = max(max(index_versions), max(data_versions))
    index_path, data_path = f"{pattern[:-1]}{max_version}.index", f"{pattern[:-1]}{max_version}.json"
    return index_path, data_path

# Borrowed (stolen) from https://stackoverflow.com/a/56253636
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--visualize", action="store_true", help="Determine whether to visualize the topic search system")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    index_path, data_path = get_index_query_files()
    index, data = load_index(index_path), load_query_data(data_path)
    model = load_model()

    while (query_str := input("Enter the Query String, or enter QUIT to quit: ")).strip().lower() != "quit":
        sentences, categories, urls, scores, misinf, idx = query_index(index, query_str, data)
        good_sentences = sentences[~misinf]
        for i in range(len(good_sentences)):
            print(f"Response {i+1}: ========================================================================")
            print(f"Article Info: {good_sentences[i]}")
            print(f"Category: {categories[~misinf][i]}")
            print(f"URL: {urls[~misinf][i]}")
            print("=========================================================================================\n")
        good_idx = idx[~misinf]
        if args.visualize:
            _, _, embeds = load_embeds(data["embed_file"]) 
            query_embed = encode_sentences(model, [query_str])
            disp_embeds = tsne_reduce_dataset(np.concatenate((embeds, query_embed)), n_components=3)

            good_disp_embeds = disp_embeds[good_idx]
            query_disp_embed = disp_embeds[-1]

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            x, y, z = disp_embeds[:-1].T
            ax.scatter(x, y, z, color="grey")
            
            for e in good_disp_embeds:
                x, y, z = np.concatenate((np.expand_dims(e, axis=0), np.expand_dims(query_disp_embed, axis=0))).T
                ax.plot(x, y, z, color="green")
                ax.scatter(x[0], y[0], z[0], color="green", label="Result")
                ax.scatter(x[1], y[1], z[1], color="blue", label="Query")
            legend_without_duplicate_labels(ax)
            plt.title("Search Result Visualization")
            plt.show()

        new_bad_string = input("Input the responses that don't seem to be correct as a comma separated list: ")
        if new_bad_string != "":
            new_bads = [int(s) - 1 for s in new_bad_string.split(',') if int(s) > 0 and int(s) <= len(good_sentences)]
            new_bad_idxs = idx[~misinf][new_bads]
            data['misinf_ratio'][new_bad_idxs] = (data['misinf_ratio'][new_bad_idxs] * data['n_queries'][new_bad_idxs] + 1) / (data['n_queries'][new_bad_idxs] + 1)
            data['n_queries'][new_bad_idxs] += 1
        
        misinf_sentences = sentences[misinf]
        if len(misinf_sentences) != 0:
            bad_idx = idx[misinf]
            support_samples = []
            for i, sent in enumerate(misinf_sentences):
                _, _, _, _, misinf, idx = query_index(index, sent, data)
                good_support_idx = idx[~misinf]
                if args.visualize:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')

                    good_disp_embeds = disp_embeds[good_support_idx]
                    doc_disp_embed = disp_embeds[bad_idx[i]]
                    x, y, z = disp_embeds[:-1].T
                    ax.scatter(x, y, z, color="grey")
            
                    for e in good_disp_embeds:
                        x, y, z = np.concatenate((np.expand_dims(e, axis=0), np.expand_dims(doc_disp_embed, axis=0))).T
                        ax.plot(x, y, z, color="green")
                        ax.scatter(x[0], y[0], z[0], color="green", label="Support")
                        ax.scatter(x[1], y[1], z[1], color="red")
                support_samples += [good_support_idx]
            if args.visualize:
                query_disp_embed = disp_embeds[-1]
                bad_disp_embeds = disp_embeds[bad_idx]
                for e in bad_disp_embeds:
                    x, y, z = np.concatenate((np.expand_dims(e, axis=0), np.expand_dims(query_disp_embed, axis=0))).T
                    ax.plot(x, y, z, color="red")
                    ax.scatter(x[0], y[0], z[0], color="red", label="Suspicious")
                    ax.scatter(x[1], y[1], z[1], color="blue", label="Query")
                legend_without_duplicate_labels(ax)
                plt.title("Misinformation Detection and Support")
                plt.show()
            support_samples = np.concatenate(support_samples)
            print("\n\nThe following queries returned potentially suspicious results:")
            for i in range(len(misinf_sentences)):
                print(f"Suspicious Response {i+1}: =============================================================")
                print(f"Article Info: {misinf_sentences[i]}")
                print(f"Category: {categories[misinf][i]}")
                print(f"URL: {urls[misinf][i]}")
                print("=========================================================================================")
            print("\n\nHere are some sources regarding these bad resources: ")
            for i in support_samples:
                sentence = data['sentences'][i]
                category = data['categories'][i]
                url = data['urls'][i]
                print("=========================================================================================")
                print(f"Article Info: {sentence}")
                print(f"Category: {category}")
                print(f"URL: {url}")
                print("=========================================================================================\n")

            new_good_string = input("With the following info, input the suspicious responses that seem to credible as a comma separated list: ")
            if new_good_string != "":
                new_goods = [int(s) - 1 for s in new_good_string.split(',') if int(s) > 0 and int(s) <= len(misinf_sentences)]
                new_good_idxs = idx[misinf][new_goods]
                data['misinf_ratio'][new_good_idxs] = max(0, data['misinf_ratio'][new_good_idxs] * data['n_queries'][new_good_idxs] - 1) / (data['n_queries'][new_good_idxs] + 1)
                data['n_queries'][new_good_idxs] += 1
            
    save_query_data(data_path, data)
    print("Saved Data Sucessfully, Thank You!")
