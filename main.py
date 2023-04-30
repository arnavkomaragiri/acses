import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from network import LatticeNetwork
from enum import Enum
from tqdm import tqdm
from typing import Tuple, Callable, Optional, List
from scipy.spatial.distance import cdist

from ant import Ant
from network import LatticeNetwork

from store_visualize import load_embeds, balance_dataset_idx
from model import load_model
from search import load_query_data

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--width", type=int, default=100)
    args.add_argument("-n", "--num-ants", type=int, default=1000)
    args.add_argument("-b", "--beta", type=float, default=32)
    args.add_argument("-d", "--delta", type=float, default=0.2)
    args.add_argument("-k", "--reinforce-exp", type=float, default=3)
    args.add_argument("-i", "--input-data", type=str, default='test_embed.pkl')
    args.add_argument("-s", "--num-steps", type=int, default=200)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.995)
    args.add_argument("-r", "--centroid-radius", type=int, default=1)
    args.add_argument("-z", "--zeros", action='store_true')
    args.add_argument("-q", "--greedy-prob", type=float, default=0.2)
    args.add_argument("-m", "--warmup-steps", type=int, default=0)
    args.add_argument("-e", "--export-video", action='store_true')
    return args.parse_args()

def find_pheromone_map(ant, pheromones, vec):
    diffs = np.zeros(pheromones.shape[:2])
    for j, row in enumerate(pheromones):
        for k, p in enumerate(row):
            diffs[j, k] = ant.find_edge_pheromone(p, vec)
    return diffs

def organize_network(network: LatticeNetwork, ants: List[Tuple[int, Ant]], embeds: np.ndarray, sents: np.ndarray,
                     num_steps: int, beta: float, delta: float, q: float, reinforce_exp: float, warmup_steps: int = 0,
                     visualize: bool = False, enc: Optional[np.ndarray] = None):
    ant_locs = []
    count = 0 # args.num_ants
    total_ages = []
    frames = []
    # run ACO self organization
    with tqdm(range(args.num_steps)) as t_iter:
        for i in t_iter:
            rng.shuffle(ants)
            sum_age = 0
            ages = []
            for u, (j, ant) in enumerate(ants):
                # new_pheromone = ant.get_new_pheromone_vec(network)
                pheromone_update = ant.get_pheromone_update_func()
                neighborhood_func = ant.get_neighborhood_func()
                network.deposit_pheromone_delta(pheromone_update, neighborhood_func, *ant.pos)
                warmup = ant.age <= warmup_steps
                s = ant.decide_next_position(network, q=q, warmup=warmup, search=True)
                if s and not warmup:
                    loc = tuple(rng.choice(np.arange(network.documents.shape[0]), 2))
                    vec = embeds[count]
                    doc = sents[count]
                    k = count
                    count = (count + 1) % len(ants)
                    # if ant.best_loc is not None and ant.pos != ant.best_loc:
                    #     network.add_edge(ant.pos, ant.best_loc)
                    #     ant.pos = ant.best_loc
                    #     network.deposit_pheromone_delta(pheromone_update, neighborhood_func, *ant.best_loc)
                    network.deposit_document(*ant.pos, ant.document, ant.vec)
                    total_ages += [ant.age]
                    ants[u] = (j, Ant(vec, loc, 1, beta, delta, reinforce_exp=reinforce_exp, ant_id=k, document=doc))
                    status[j] = False
                else:
                    status[j] = s
                sum_age += ant.age
                ages += [ant.age]
            network.evaporate_pheromones()
            if i % 50 == 49:
                network.erode_network(min_dist=0.7)
            norms = np.linalg.norm(network.pheromones, axis=-1)
            best_matches = [ant.current_pheromone for _, ant in ants]
            t_iter.set_postfix(avg_pheromone_norm=np.mean(norms), avg_age=np.mean(ages), min_age=np.min(ages), max_age=np.max(ages), 
                               best_match=np.max(best_matches), avg_match=np.mean(best_matches), count=count)
            if visualize:
                map = find_pheromone_map(ant, network.pheromones, enc)
                frames.append([plt.imshow(map, animated=True)])
    if visualize:
        return network, ants, ages, total_ages, frames 
    return network, ants, ages, total_ages

def init_ant(network: LatticeNetwork, vec: np.ndarray, beta: float, delta: float, doc: str = "", verbose: bool = False, rng = None) -> Ant:
    if rng is None:
        rng = np.random
    new_pos = tuple(rng.choice(np.arange(network.documents.shape[0]), 2))
    new_ant = Ant(vec, new_pos, 1, beta, delta, document=doc)
    start_match = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
    if verbose:
        print(f"Start Position: {new_ant.pos}, Start Match: {start_match}")
    return new_ant

def ant_search(network: LatticeNetwork, ant: Ant, q: float, max_steps: Optional[int] = None):
    pos_seq = []
    pheromone_seq = []
    prev_status = False
    i = 0

    while True:
        if max_steps is not None and i > max_steps:
            return None
        pos_seq += [ant.pos]
        status = ant.decide_next_position(network, q=q, search=True)
        pheromone = ant.find_edge_pheromone(network.get_pheromone_vec(*ant.pos), ant.vec)
        # if len(pheromone_seq) != 0 and pheromone < pheromone_seq[-1]:
            # status = True
        pheromone_seq += [pheromone]
        # print(status, pheromone / np.max(diffs[new_ant_class]))
        # if pheromone > (0.6 * np.max(diffs[new_ant_class])) and status:
        # stop if we get two consecutive stop signals
        a, b = ant.pos
        if status and len(network.documents[a, b]) != 0:
            return ant, network.documents[a, b], pos_seq, pheromone_seq
        prev_status = status
        i += 1


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    args = parse_args()
    model = load_model("all-mpnet-base-v2")

    if args.export_video:
        sent = input("Input Query String: ")
        enc = model.encode(sent)
        fig = plt.figure()
        frames = []

    categories, sentences, embeddings = load_embeds(args.input_data)
    idxs = balance_dataset_idx(categories, 8 * args.num_ants, rng=rng)
    # e = embeddings[idxs]
    # dists = cdist(e, e)
    # plt.imshow(dists)
    # plt.show()

    # dots = 1 - (e @ e.T)
    # plt.imshow(dots)
    # plt.show()

    sents = sentences[idxs]
    embeds = embeddings[idxs]

    network = LatticeNetwork((args.width, args.width), embeds.shape[-1], args.evaporation_factor, 
                             rng=rng, centroid_radius=args.centroid_radius, zeros=args.zeros)
    existing_locs = set()
    ants = []
    status = []
    for i in range(args.num_ants):
        ant_vec = embeds[i]
        loc = tuple(rng.choice(np.arange(args.width), 2))
        ants += [(i, Ant(ant_vec, loc, 1, args.beta, args.delta, reinforce_exp=args.reinforce_exp, ant_id=i, document=sents[i]))]
        status += [False]

    # ant_locs = []
    # count = 0 # args.num_ants
    # total_ages = []
    # # run ACO self organization
    # with tqdm(range(args.num_steps)) as t_iter:
    #     for i in t_iter:
    #         rng.shuffle(ants)
    #         sum_age = 0
    #         ages = []
    #         for u, (j, ant) in enumerate(ants):
    #             # new_pheromone = ant.get_new_pheromone_vec(network)
    #             pheromone_update = ant.get_pheromone_update_func()
    #             neighborhood_func = ant.get_neighborhood_func()
    #             network.deposit_pheromone_delta(pheromone_update, neighborhood_func, *ant.pos)
    #             s = ant.decide_next_position(network, args.greedy_prob)
    #             if s and status[j] and i > args.warmup_steps:
    #                 loc = tuple(rng.choice(np.arange(args.width), 2))
    #                 vec = embeds[count]
    #                 doc = sents[count]
    #                 k = count
    #                 count = (count + 1) % args.num_ants
    #                 # if ant.best_loc is not None and ant.pos != ant.best_loc:
    #                 #     network.add_edge(ant.pos, ant.best_loc)
    #                 #     ant.pos = ant.best_loc
    #                 #     network.deposit_pheromone_delta(pheromone_update, neighborhood_func, *ant.best_loc)
    #                 network.deposit_document(*ant.pos, ant.document, ant.vec)
    #                 total_ages += [ant.age]
    #                 ants[u] = (j, Ant(vec, loc, 1, args.beta, args.delta, reinforce_exp=args.reinforce_exp, ant_id=k, document=doc))
    #                 status[j] = False
    #             else:
    #                 status[j] = s
    #             sum_age += ant.age
    #             ages += [ant.age]
    #         network.evaporate_pheromones()
    #         if i % 50 == 49:
    #             network.erode_network()
    #         norms = np.linalg.norm(network.pheromones, axis=-1)
    #         best_matches = [ant.best_pheromone for _, ant in ants]
    #         t_iter.set_postfix(avg_pheromone_norm=np.mean(norms), avg_age=np.mean(ages), min_age=np.min(ages), max_age=np.max(ages), 
    #                            best_match=np.max(best_matches), avg_match=np.mean(best_matches), count=count)

    #         if args.export_video:
    #             map = find_pheromone_map(ant, network.pheromones, enc)
    #             frames.append([plt.imshow(map, animated=True)])
    #         # t_iter.set_postfix(num_stopped=sum(status))
    #         # pct_stop = sum(status) / len(status)
    #         # if pct_stop > 0.9:
    #         #     break
    if args.export_video:
        network, ants, ages, total_ages, frames = organize_network(network, ants, embeds, sents, args.num_steps, 
                                                                   args.beta, args.delta, args.greedy_prob, args.reinforce_exp,
                                                                   args.warmup_steps, args.export_video, enc)
        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
        ani.save("animation.mp4")
        plt.show()
    else:
        network, ants, ages, total_ages = organize_network(network, ants, embeds, sents, args.num_steps, 
                                                           args.beta, args.delta, args.greedy_prob, args.reinforce_exp,
                                                           args.warmup_steps)

    total_ages += ages
    plt.hist(total_ages, bins=np.ptp(total_ages)+1)
    plt.title("Ant Age Histogram")
    plt.show()

    unique, counts = np.unique(total_ages, return_counts=True)
    log_unique = np.log(unique[1:])
    log_freqs = np.log(counts[1:] / np.sum(counts[1:]))
    plt.plot(log_unique, log_freqs)
    plt.title("Ant Age Rank Plot")
    plt.show()

    lens = [len(x) for x in network.neighbors.flatten()]
    u, c = np.unique(lens, return_counts=True)
    lu, lc = np.log(u), np.log(c / np.sum(c))
    plt.loglog(u, c)
    plt.title("Network Degree Rank Plot")
    plt.show()

    lens = [len(x) for x in network.documents.flatten()]
    u, c = np.unique(lens, return_counts=True)
    lu, lc = np.log(u), np.log(c / np.sum(c))
    plt.loglog(u, c)
    plt.title("Document Count Rank Plot")
    plt.show()

    i = np.argmax(ages)
    vec = ants[i][1].vec
    diffs = find_pheromone_map(ants[i][1], network.pheromones, vec)
    print(f"Sentence: {sents[ants[i][1].ant_id]}")
    plt.imshow(diffs)
    plt.show()

    sentence = input("Input Query String: ")
    emb = model.encode(sentence)
    diffs2 = find_pheromone_map(ants[i][1], network.pheromones, emb)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(diffs)
    ax[1].imshow(diffs2)
    plt.show()

    new_ant = init_ant(network, emb, args.beta, args.delta, doc=sentence, verbose=True)
    new_ant, docs, pos_seq, pheromone_seq = ant_search(network, new_ant, args.greedy_prob)

    print("Search Results: ")
    for d in docs:
        print(d)

    final_match = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
    print(f"Path Length: {len(pos_seq)}")
    print(f"Final Position: {new_ant.pos}, Final Match: {final_match}")

    path_map = np.zeros((args.width, args.width))
    path_diff = find_pheromone_map(new_ant, network.pheromones, new_ant.vec)
    for i, (r, c) in enumerate(pos_seq):
        p = new_ant.find_edge_pheromone(network.pheromones[r, c], new_ant.vec)
        path_map[r, c] = p / np.max(path_diff)
    plt.imshow(path_map)
    plt.show()

    out_path = input("Input Output Path: ")
    try:
        network.to_pickle(out_path)
        n = LatticeNetwork.from_pickle(out_path)
    except Exception as e:
        print(f"failed to save network state to pickle: {str(e)}")