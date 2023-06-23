import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from network import LatticeNetwork, HierarchicalLattice, euclidean_dist, ip_dist, inv_cos_dist
from enum import Enum
from tqdm import tqdm
from typing import Tuple, Callable, List, Optional

rng = np.random.default_rng(seed=0)

AntState = Enum('AntState', ['FOLLOW', 'RECOVER', 'STOPPED'])


class Ant:
    def __init__(self, vec: np.ndarray, pos: tuple, alpha: float, beta: float, delta: float, 
                 eps: float = 0.01, reinforce_exp: float = 3.0, ant_id: int = None, document: str = None,
                 geometry: str = "euclidean"):
        # initialize ant with the document vector
        self.vec = vec
        self.pos = pos
        self.prev_pos = None
        self.eps = eps
        self.geometry = geometry

        # initialize hyperparams
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.reinforce_exp = reinforce_exp

        # initialize ant state and memory
        self.state = AntState.FOLLOW
        self.best_pheromone = 0
        self.current_pheromone = 0
        self.best_loc = None
        self.age = 0
        self.pos_seq = []
        self.pher_seq = []
        self.deg_seq = []

        # initialize distance function
        self.dist = ip_dist

        # initialize tracking
        self.ant_id = ant_id
        # initialize document
        self.document = document

    def get_move_diff(self, candidate: Tuple, width: int):
        if self.geometry == "euclidean":
            if self.prev_pos is None:
                return 1.0
        
            curr_move = np.array(candidate) - np.array(self.pos)
            prev_move = np.array(self.pos) - np.array(self.prev_pos)
            if (np.abs(curr_move) > (width / 2)).any():
                idx = (np.abs(curr_move) > (width / 2)).nonzero()[0]
                inv_sgn = np.sign(curr_move[idx])
                curr_move[idx] = inv_sgn * (width - np.abs(curr_move[idx]))
            if (np.abs(prev_move) > (width / 2)).any():
                idx = (np.abs(prev_move) > (width / 2)).nonzero()[0]
                inv_sgn = np.sign(prev_move[idx])
                prev_move[idx] = inv_sgn * (width - np.abs(prev_move[idx]))
        
            curr_move = curr_move / (np.linalg.norm(curr_move) + 0.0001)
            prev_move = prev_move / (np.linalg.norm(prev_move) + 0.0001)
            angle = np.arccos(np.clip(np.dot(curr_move, prev_move), -1.0, 1.0)) 

            return max(1 - (angle / np.pi), 0.5)
        elif self.geometry == "graph":
            num_visited = self.pos_seq.count(candidate)
            penalty = 2 ** (-num_visited)
            return penalty

    def get_pheromone_landscape(self, network: LatticeNetwork, r: int = 1, pos: Optional[Tuple] = None, exclude_list: List = []) -> np.ndarray:
        if pos is None:
            pos = self.pos
        # compute neighbors and corresponding pheromone levels
        neighbors = network.get_neighborhood(*pos, radius=r, exclude_list=exclude_list)
        centroid_vecs = [network.get_centroid_pheromone_vec(r, c, [pos]) for r, c in neighbors]
        pheromones = [self.find_edge_pheromone(centroid_vecs[i], self.vec) for i in range(len(neighbors))]
        
        # scale decisions based on document count heatmaps
        if network.count_heatmap is not None:
            dists = [self.dist(centroid_vecs[i], self.vec) for i in range(len(neighbors))]
            count_heats = [network.count_heatmap[r, c] for r, c in neighbors]
            count_mults = [h ** (0.5 * (2 - d)) for h, d in zip(count_heats, dists)]
            pheromones = [count_mults[i] * p for i, p in enumerate(pheromones)]
        return neighbors, pheromones

    def advance(self, network: LatticeNetwork, q: float = 0.2, r: int = 1, exclude_list: List = [],
                warmup: bool = False, search: bool = False, trim_pct: float = 0.0, move_diff: bool = True) -> bool:
        # get local pheromone landscape
        neighbors, pheromones = self.get_pheromone_landscape(network, r, exclude_list=exclude_list+[self.pos])

        # compute current node pheromones
        cent = network.get_centroid_pheromone_vec(*self.pos)
        self.current_pheromone = self.find_edge_pheromone(cent, self.vec)

        # enforce that pheromones consistently get better if there is room to improve
        if search and any([p >= self.current_pheromone for p in pheromones]):
            pheromones = [p if p >= self.current_pheromone else 0 for p in pheromones]
        
        if trim_pct != 0.0:
            for _ in range(int(trim_pct * len(pheromones))):
                i = np.argmin(pheromones)
                pheromones[i] = 0

        # compute the phermone scalars for the change in directions
        if move_diff:
            move_diffs = [self.get_move_diff(n, network.pheromones.shape[0]) for n in neighbors]
            pheromones = [move_diffs[i] * p for i, p in enumerate(pheromones)]
        pheromones += [self.current_pheromone]

        if self.current_pheromone > self.best_pheromone:
            self.best_pheromone = self.current_pheromone
            self.best_loc = self.pos

        probs = np.array(pheromones) / np.sum(pheromones)
        
        stopped = False
        if rng.uniform() < q:
            # take the greedy option with probability q
            i = np.argmax(pheromones)
            j = np.argmax(pheromones[:-1])
        else:
            i = int(self.roulette_wheel(probs))
            j = int(self.roulette_wheel(pheromones[:-1] / np.sum(pheromones[:-1])))

        stopped = (i == len(pheromones) - 1)
        if stopped:
            i = j

        if not stopped or warmup:
            new_pos = neighbors[i]
            # TODO: Implement previous move tracking
            self.prev_pos = self.pos
            self.pos = new_pos
            self.age += 1

            self.pos_seq += [self.pos]
            self.pher_seq += [self.current_pheromone]
            self.deg_seq += [len(neighbors)]
        return stopped and not warmup

    def get_rewire_pos(self, network: LatticeNetwork, m: int, r: int = 1) -> List[Tuple[int, int]]:
        # find gateway position
        gateway_pos = self.sample_gateway_pos()
        # get neighborhood and corresponding pheromones
        gateway_neighborhood = network.get_neighborhood(*gateway_pos, r)
        gateway_centroids = [network.get_centroid_pheromone_vec(r, c) for r, c in gateway_neighborhood]
        pheromones = [self.find_edge_pheromone(c, self.vec) for c in gateway_centroids] 
        # convert pheromones to probabilities
        probs = np.array(pheromones) / np.sum(pheromones)
        idxs = self.roulette_wheel(probs, num_samples=m, replace=False)
        return [gateway_neighborhood[i] for i in idxs]

    def get_local_rewire_pos(self, network: LatticeNetwork, m: int, r: int = 1):
        gateways = network.get_neighborhood(*self.pos, r)
        degs = [len(network.neighbors[a, b]) for a, b in gateways]
        probs = np.array(degs) / np.sum(degs)
        i = int(self.roulette_wheel(probs, num_samples=1))

        neighbors = network.get_neighborhood(*gateways[i], r)
        centroids = [network.get_centroid_pheromone_vec(r, c) for r, c in neighbors]
        pheromones = [self.find_edge_pheromone(c, self.vec) for c in centroids]

        idxs = self.roulette_wheel(np.array(pheromones) / np.sum(pheromones), num_samples=m, replace=False)
        return [neighbors[i] for i in idxs]

    def sample_gateway_pos(self) -> Tuple[int, int]:
        if len(self.deg_seq) == 0:
            raise ValueError("cannot find gateway node unless ant has traversed at least one node")
        mod_seq = np.array(self.deg_seq) - 8 + self.eps
        probs = mod_seq / np.sum(mod_seq)
        i = int(self.roulette_wheel(probs))
        return self.pos_seq[i]

    # do a roulette wheel decision with weighted probabilities
    def roulette_wheel(self, probs: np.ndarray, num_samples: int = 1, **kwargs) -> float:
        return rng.choice(np.arange(len(probs)), num_samples, p=probs, **kwargs)

    # apply ant pheromone weighting
    def pheromone_weighting(self, sigma: float) -> float:
        # TODO: are we doing the ant path smoothing factor here? See equation (9) in Fernandes et al.
        return (1 + (self.delta / (1 + (sigma * self.delta)))) ** self.beta

    # compute the pheromone of the edge between two nodes (equation (10) in Fernandes et al.)
    def find_edge_pheromone(self, centroid_pheromone: np.ndarray, node_pheromone: np.ndarray) -> float:
        unweight_pheromone = self.dist(centroid_pheromone, node_pheromone, self.eps)
        return self.pheromone_weighting(unweight_pheromone)

    # get pheromone update reinforce (R in equation (12) in Fernandes et al.)
    def get_update_reinforce(self, centroid_pheromone: np.ndarray) -> float:
        # TODO: do we need to divide by the number of variables in the node pheromone if all our vectors are normalized?
        d = self.dist(centroid_pheromone, self.vec, self.eps) / 2.0
        return self.alpha * ((1 - d) ** self.reinforce_exp)

    # update the pheromone vector for the specified node (equation (11) in Fernandes et al.)
    def get_new_pheromone_vec(self, network: LatticeNetwork) -> np.ndarray:
        centroid_pheromone = network.get_centroid_pheromone_vec(*self.pos)
        node_pheromone = network.get_pheromone_vec(*self.pos)
        reinforce = self.get_update_reinforce(centroid_pheromone)
        vec = node_pheromone + reinforce * (self.vec - node_pheromone)
        return vec

    def get_pheromone_update_func(self):
        def update(centroid: np.ndarray, node: np.ndarray, alpha: float = 1.0) -> float:
            reinforce = self.get_update_reinforce(centroid)
            vec = node + alpha * reinforce * (self.vec - node)
            return vec
        return update

    def get_match_func(self):
        def match(centroid_pheromone: np.ndarray) -> float:
            return self.find_edge_pheromone(centroid_pheromone, self.vec)
        return match

    def get_neighborhood_func(self):
        def func(dist: float) -> float:
            return (0.25) ** (dist)
        return func

class HierarchicalAnt(Ant):
    def __init__(self, vec: np.ndarray, pos: tuple, alpha: float, beta: float, delta: float, level: int = 0,
                 eps: float = 0.01, reinforce_exp: float = 3.0, ant_id: int = None, document: str = None):
        super().__init__(vec, pos, alpha, beta, delta, eps, reinforce_exp, ant_id, document)
        self.level = level

    def advance(self, network: HierarchicalLattice, 
                q: float = 0.2, r: int = 1, warmup: bool = False, search: bool = True, trim_pct: float = 0) -> bool:
        net = network.get_level_network(self.level)
        # find the warmup threshold, and update our warmup flag if we are using warmup
        warmup_thresh = network.widths[self.level] // 4
        warmup = warmup and self.age <= warmup_thresh
        # get our stop signal from 
        stop = super().advance(net, q, r, warmup, search, trim_pct)

        if stop and self.level < len(network.levels) - 1:
            self.level += 1
            self.age = 0
            self.pos = network.get_next_level_pos(*self.pos, self.level)
        elif stop:
            return True
        return False
        


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-n", "--num-ants", type=int, default=5)
    args.add_argument("-w", "--width", type=int, default=5)
    args.add_argument("-r", "--centroid-radius", type=int, default=1)
    args.add_argument("-b", "--beta", type=float, default=32)
    args.add_argument("-d", "--delta", type=float, default=0.2)
    args.add_argument("-e", "--embedding-dim", type=int, default=5)
    args.add_argument("-c", "--num-classes", type=int, default=5)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.9925)
    args.add_argument("-s", "--num-steps", type=int, default=600)
    args.add_argument("-m", "--warmup-steps", type=int, default=200)
    args.add_argument("-q", "--greedy-prob", type=float, default=0.2)
    args.add_argument("-z", "--zeros", action='store_true')
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    network = LatticeNetwork((args.width, args.width), args.embedding_dim, args.evaporation_factor, 
                             rng=rng, centroid_radius=args.centroid_radius, zeros=args.zeros)
    existing_locs = set()
    ants, status = [], []
    vec_set = np.eye(args.num_classes, args.embedding_dim)
    ant_set = [[] for _ in range(args.num_classes)]
    for i in range(args.num_ants):
        j = rng.choice(args.num_classes)
        ant_vec = vec_set[j] + rng.normal(scale=0.2, size=args.embedding_dim)
        while True:
            loc = tuple(rng.choice(np.arange(args.width), 2))
            if loc not in existing_locs or True:
                existing_locs.add(loc)
                break
        ants += [(i, Ant(ant_vec, loc, 1, args.beta, args.delta))]
        ant_set[j] += [i]
        status += [False]

    for i, s in enumerate(status):
        print(f"Starting Status {i}: {status[i]}, Pos: {ants[i][1].pos}")

    # run ACO self organization
    with tqdm(range(args.num_steps)) as t_iter:
        for i in t_iter:
            rng.shuffle(ants)
            for j, ant in ants:
                # new_pheromone = ant.get_new_pheromone_vec(network)
                pheromone_update = ant.get_pheromone_update_func()
                neighborhood_func = ant.get_neighborhood_func()
                network.deposit_pheromone_delta(pheromone_update, neighborhood_func, *ant.pos)
                if not status[j] and i > args.warmup_steps:
                    status[j] = ant.decide_next_position(network, args.greedy_prob)
                elif i <= args.warmup_steps:
                    ant.decide_next_position(network, args.greedy_prob)
            network.evaporate_pheromones()
            t_iter.set_postfix(num_stopped=sum(status))
            pct_stop = sum(status) / len(status)
            if pct_stop > 0.9:
                break

    for i, ant in ants:
        final_pheromone_vec = network.get_pheromone_vec(*ant.pos)
        pheromone = ant.find_edge_pheromone(final_pheromone_vec, ant.vec)
        print(f"Status {i}: {status[i]}, Pos: {ant.pos}, Final Pheromone: {pheromone}")

    diffs = [ants[0][1].pheromone_weighting(np.linalg.norm(network.pheromones - vec, axis=-1)) for vec in vec_set]
    min_diffs = [np.min(diff) for diff in diffs]
    for j, ant_idxs in enumerate(ant_set):
        class_ant_pos = [ant.pos for idx, ant in ants if idx in ant_idxs]
        for r, c in class_ant_pos:
            diffs[j][r, c] = min_diffs[j] / (2 ** class_ant_pos.count((r, c)))

    fig, ax = plt.subplots(1, args.num_classes) 
    for i, diff in enumerate(diffs):
        ax[i].imshow(diff)
    plt.show()

    new_ant_class = int(input("Provide New Ant Class: "))
    new_pos = tuple(map(int, input("Input the Start Position as a Comma Separated Point: ").split(",")))
    new_ant = Ant(vec_set[new_ant_class], new_pos, 1, args.beta, args.delta)
    start_match = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
    print(f"Start Position: {new_ant.pos}, Start Match: {start_match}")

    pos_seq = []
    pheromone_seq = []
    prev_status = False
    while True:
        pos_seq += [new_ant.pos]
        status = new_ant.advance(network, q=args.greedy_prob)
        pheromone = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
        pheromone_seq += [pheromone]
        # print(status, pheromone / np.max(diffs[new_ant_class]))
        # if pheromone > (0.6 * np.max(diffs[new_ant_class])) and status:
        # stop if we get two consecutive stop signals
        if status and prev_status:
            break
        prev_status = status

    final_match = new_ant.find_edge_pheromone(network.get_pheromone_vec(*new_ant.pos), new_ant.vec)
    print(f"Path Length: {len(pos_seq)}")
    print(f"Final Match: {final_match}")
    path_diff = new_ant.pheromone_weighting(np.linalg.norm(network.pheromones - vec_set[new_ant_class], axis=-1))
    path_map = np.zeros((args.width, args.width))
    for i, (r, c) in enumerate(pos_seq):
        p = new_ant.pheromone_weighting(np.linalg.norm(network.get_pheromone_vec(r, c) - new_ant.vec))
        path_map[r, c] = p / np.max(path_diff)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(path_diff)
    ax[1].imshow(path_map)
    ax[2].plot(pheromone_seq)
    plt.show()