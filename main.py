import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from network import LatticeNetwork
from enum import Enum
from tqdm import tqdm
from typing import Tuple, Callable
from scipy.spatial.distance import cdist

from ant import Ant
from network import LatticeNetwork

from store_visualize import load_embeds
from search import load_query_data

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--width", type=int)
    args.add_argument("-n", "--num-ants", type=int)
    args.add_argument("-b", "--beta", type=float, default=32)
    args.add_argument("-d", "--delta", type=float, default=0.2)
    args.add_argument("-i", "--input-data", type=str)
    args.add_argument("-s", "--num-steps", type=int, default=200)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.995)
    args.add_argument("-r", "--centroid-radius", type=int, default=1)
    args.add_argument("-z", "--zeros", action='store_true')
    args.add_argument("-q", "--greedy-prob", type=float, default=0.2)
    return args.parse_args()

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    args = parse_args()

    categories, sentences, embeds = load_embeds(args.input_data)
    e = embeds[:1000]
    dists = cdist(e, e)
    plt.imshow(dists)
    plt.show()

    dots = 1 - (e @ e.T)
    plt.imshow(dots)
    plt.show()

    network = LatticeNetwork((args.width, args.width), embeds.shape[-1], args.evaporation_factor, 
                             rng=rng, centroid_radius=args.centroid_radius, zeros=args.zeros)
    existing_locs = set()
    ants = []
    status = []
    for i in range(args.num_ants):
        ant_vec = embeds[i]
        loc = tuple(rng.choice(np.arange(args.width), 2))
        ants += [(i, Ant(ant_vec, loc, 1, args.beta, args.delta))]
        status += [False]

    count = args.num_ants
    # run ACO self organization
    with tqdm(range(args.num_steps)) as t_iter:
        for i in t_iter:
            rng.shuffle(ants)
            sum_age = 0
            ages = []
            for j, ant in ants:
                # new_pheromone = ant.get_new_pheromone_vec(network)
                pheromone_update = ant.get_pheromone_update_func()
                neighborhood_func = ant.get_neighborhood_func()
                network.deposit_pheromone_delta(pheromone_update, neighborhood_func, *ant.pos)
                s = ant.decide_next_position(network, args.greedy_prob)
                if s and status[j]:
                    loc = tuple(rng.choice(np.arange(args.width), 2))
                    ants[j] = (j, Ant(embeds[count], loc, 1, args.beta, args.delta))
                    status[j] = False
                else:
                    status[j] = s
                sum_age += ant.age
                ages += [ant.age]
            network.evaporate_pheromones()
            norms = np.linalg.norm(network.pheromones, axis=-1)
            best_matches = [ant.best_pheromone for j, ant in ants]
            t_iter.set_postfix(avg_pheromone_norm=np.mean(norms), avg_age=np.mean(ages), min_age=np.min(ages), max_age=np.max(ages), 
                               best_match=np.max(best_matches))
            # t_iter.set_postfix(num_stopped=sum(status))
            # pct_stop = sum(status) / len(status)
            # if pct_stop > 0.9:
            #     break
    plt.hist(ages, bins=np.ptp(ages)+1)
    plt.show()

    i = np.argmax(ages)
    diffs = np.zeros(network.pheromones.shape[:2])
    for j, row in enumerate(network.pheromones):
        for k, p in enumerate(row):
            diffs[j, k] = ants[i][1].pheromone_weighting(ants[i][1].dist(p, ants[i][1].vec))
    # pheromone_list = network.pheromones.reshape(-1, network.pheromones.shape[-1])
    # diff = ants[i][1].pheromone_weighting(ants[i][1].dist(pheromone_list, ants[i][1].vec))
    # diff = ants[i][1].pheromone_weighting(np.linalg.norm(network.pheromones - ants[i][1].vec, axis=-1))
    plt.imshow(diffs)
    plt.show()

    
