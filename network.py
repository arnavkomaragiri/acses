import pickle

import numpy as np

from numpy.random import Generator
from scipy.stats import norm
from scipy.special import zeta

from functools import reduce
from typing import List, Tuple, Union, Optional, Callable

def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def inv_cos_dist(a: np.ndarray, b: np.ndarray, eps: float = 0.001) -> float:
    a_norm, b_norm = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    return 1 - ((np.dot(a, b)) / ((a_norm * b_norm) + eps))

def ip_dist(a: np.ndarray, b: np.ndarray, eps: float = 0.001) -> float:
    return 1 - np.dot(a, b)

class LatticeNetwork():
    def __init__(self, network_shape: Tuple, embedding_dimension: int, evap_factor: float, 
                 centroid_radius: int = 1, rng: Optional[Generator] = None, zeros: bool = False):
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        # initialize centroid radius and evaporation factor
        self.centroid_radius = centroid_radius
        self.evap_factor = evap_factor

        if type(network_shape) != tuple:
            raise ValueError('network_shape should be a numpy shape (i.e. a tuple)')
    
        if type(embedding_dimension) != int:
            raise ValueError('embedding_dimension should be an int')

        if rng is None and not zeros:
            # initialize normally distributed pheromone vectors
            unnormalized_pheromones = np.random.randn(network_shape[0], network_shape[1], embedding_dimension)
            # normalize each node's pheromone vector to an L2-norm of 1 to project to spherical points
            self.pheromones = unnormalized_pheromones / np.linalg.norm(unnormalized_pheromones, axis=-1, keepdims=True)
        elif not zeros:
            unnormalized_pheromones = rng.normal(size=(network_shape[0], network_shape[1], embedding_dimension))
            # normalize each node's pheromone vector to an L2-norm of 1 to project to spherical points
            self.pheromones = unnormalized_pheromones / np.linalg.norm(unnormalized_pheromones, axis=-1, keepdims=True)
        else:
            self.pheromones = np.zeros((network_shape[0], network_shape[1], embedding_dimension))
        self.init_pheromones = np.copy(self.pheromones)

        # initialize document list
        self.documents = np.ndarray(network_shape, dtype=list)
        self.doc_vecs = np.ndarray(network_shape, dtype=list)
        self.count_heatmap = None

        # initialize 8-neighbor neighborhood adjacency list
        self.neighbors = np.ndarray(network_shape, dtype=list)
        for row in range(network_shape[0]):
            for col in range(network_shape[1]):
                self.neighbors[row, col] = []
                self.documents[row, col] = []
                self.doc_vecs[row, col] = []
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if a == 0 and b == 0:
                            continue 
                        self.neighbors[row, col] += [((row + a) % self.neighbors.shape[0], (col + b) % self.neighbors.shape[1])]

    def mean_with_mutation(self, vecs: List[np.ndarray], dtheta: float) -> np.ndarray:
        mean_vec = np.mean(vecs, axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 0.00001)

        rand_vec = self.rng.normal(size=mean_vec.shape)
        ortho_vec = rand_vec - ((np.dot(mean_vec, rand_vec) / np.dot(mean_vec, mean_vec)) * mean_vec)
        ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)

        theta = dtheta * self.rng.random()
        gamma, delta = np.cos(theta), np.sin(theta)
        return (gamma * mean_vec) + (delta * ortho_vec)

    def smooth_init_pheromones(self, populate_pct: float, dtheta: float, k: float = 0.3):
        self.pheromones = np.zeros_like(self.pheromones)
        num_populate = int(populate_pct * reduce(lambda a, b: a * b, self.neighbors.shape))
        # initialize pos set
        pos_set = set()
        norms = np.zeros((self.pheromones.shape[0], self.pheromones.shape[1]), dtype=float)
        for _ in range(num_populate):
            # generate new position without repeated elements
            pos = tuple(self.rng.choice(self.neighbors.shape[0], 2))
            if pos in pos_set:
                continue
            # add position to pos set and randomly create vector
            pos_set.add(pos)
            r, c = pos
            norms[r, c] = 1.0
            vec = self.rng.normal(size=self.pheromones.shape[-1])
            self.pheromones[r, c] = vec / np.linalg.norm(vec)
        
        # convert pos set to open list
        open_list = reduce(lambda a, b: a + b, [self.neighbors[r, c] for r, c in pos_set])
        # run BFS to smoothly populate pheromones
        while len(open_list) != 0:
            # pop first entry off queue and get neighbors
            open_pos = open_list.pop(0)
            r, c = open_pos
            neighbors = self.neighbors[r, c]

            # get all nonzero neighboring pheromone vectors
            vecs = [self.pheromones[a, b] for a, b in neighbors if (a, b) in pos_set]

            # if there are no empty neighbors to populate, just break out of this iteration
            if len(vecs) == 0:
                continue

            # populate new pheromone via mean with mutation algo
            vec = self.mean_with_mutation(vecs, dtheta)
            self.pheromones[r, c] = self.mean_with_mutation(vecs, dtheta)

            pos_set.add(open_pos)
            open_list += [n for n in neighbors if n not in pos_set]

    def get_pheromone_vec(self, row: Union[int, np.ndarray], col: Union[int, np.ndarray]) -> np.ndarray:
        return self.pheromones[row, col]

    # TODO: do we want to enforce that the mean pheremone vectors are normalized?
    # tentative argument for not normalizing, bc the norm of a vector is indicative of the "strength" of a given topic
    # (eg. a "distracted" node will have a pheremone vector with a lower norm, which means even if it finds a match it's less likely to pick it)
    # This will result in distracted nodes being less likely to be traversed, with specialized nodes being picked instead
    def get_centroid_pheromone_vec(self, row: int, col: int, exclude_list : List[Tuple] = []) -> np.ndarray:
        region_points = np.array(self.get_neighborhood(row, col, self.centroid_radius, exclude_list=exclude_list))
        region_pheromones = self.get_pheromone_vec(*region_points.T)
        return np.mean(region_pheromones, axis=0)
    
    def get_neighborhood(self, row: int, col: int, radius: int, exclude_list: List[Tuple] = []) -> List[Tuple]:
        exclude_set = set(exclude_list)
        region_set = set()
        outer_points = self.get_neighbors(row, col)

        region_set.update(outer_points)
        if (row, col) not in exclude_set:
            region_set.add((row, col))

        for _ in range(radius - 1):
            # convert outer points being considered to a numpy array and get all the possible neighbors
            np_outer = np.array(outer_points)
            candidate_points = reduce(lambda a, b: a + b, self.get_neighbors(*np_outer.T))
            # filter candidate points to the new outer point set based on what hasn't been seen
            outer_points = [p for p in candidate_points if tuple(p) not in region_set and tuple(p) not in exclude_set]
            region_set.update(outer_points)
        return list(region_set)
   
    def get_neighbors(self, row: Union[int, np.ndarray], col: Union[int, np.ndarray]) -> Union[List[Tuple], np.ndarray]:
        return self.neighbors[row, col]

    def add_edge(self, pos1: Tuple, pos2: Tuple):
        a1, b1 = pos1
        a2, b2 = pos2
        if pos2 not in self.neighbors[a1, b1]:
            self.neighbors[a1, b1] += [pos2]
        if pos1 not in self.neighbors[a2, b2]:
            self.neighbors[a2, b2] += [pos1]

    def remove_edge(self, pos1: Tuple, pos2: Tuple):
        a1, b1 = pos1
        a2, b2 = pos2
        if pos2 not in self.neighbors[a1, b1] or pos1 not in self.neighbors[a2, b2]:
            raise ValueError(f"edge between {pos1} and {pos2} does not exist")
        self.neighbors[a1, b1].remove(pos2)
        self.neighbors[a2, b2].remove(pos1)

    # do a roulette wheel decision with weighted probabilities
    def roulette_wheel(self, probs: np.ndarray, num_samples: int = 1, **kwargs) -> float:
        return self.rng.choice(np.arange(len(probs)), num_samples, p=probs, **kwargs)

    def global_st_rewire(self, dist: Callable = inv_cos_dist, m: int = 1):
        len_fun = np.vectorize(lambda a: len(a))
        len_map = len_fun(self.documents)
        nonempty_r, nonempty_c = len_map.nonzero()

        for r, c in zip(nonempty_r, nonempty_c):
            deg_map = len_fun(self.neighbors) - 7.999999999
            for a, b in self.get_neighborhood(r, c, 1):
                deg_map[a, b] = 0
            degs = deg_map.flatten()
            probs = degs / np.sum(degs)
            i = int(self.roulette_wheel(probs, 1))
            gateway_r, gateway_c = np.unravel_index(i, deg_map.shape)

            neighbors = self.get_neighborhood(gateway_r, gateway_c, self.centroid_radius)
            pheromones = [self.get_pheromone_vec(a, b) for a, b in neighbors]

            vec = self.get_pheromone_vec(r, c)
            dists = [2 - dist(vec, p) for p in pheromones]
            sample_probs = np.array(dists) / np.sum(dists)
            idxs = self.roulette_wheel(sample_probs, num_samples=m, replace=False)
            
            for i in idxs:
                a, b = neighbors[i]
                self.add_edge((r, c), (a, b))

    def local_st_rewire(self, dist: Callable = inv_cos_dist, m: int = 1, explore_rad: int = 3):
        len_fun = np.vectorize(lambda a: len(a))
        len_map = len_fun(self.documents)
        nonempty_r, nonempty_c = len_map.nonzero()

        for r, c in zip(nonempty_r, nonempty_c):
            local_cands = self.get_neighborhood(r, c, explore_rad)
            degs = np.array([len(self.neighbors[a, b]) for a, b in local_cands]) - 7.999999999
            probs = degs / np.sum(degs)
            i = int(self.roulette_wheel(probs, 1))
            gateway_r, gateway_c = local_cands[i]

            neighbors = self.get_neighborhood(gateway_r, gateway_c, self.centroid_radius)
            pheromones = [self.get_pheromone_vec(a, b) for a, b in neighbors]

            vec = self.get_pheromone_vec(r, c)
            dists = [2 - dist(vec, p) for p in pheromones]
            sample_probs = np.array(dists) / np.sum(dists)
            idxs = self.roulette_wheel(sample_probs, num_samples=m, replace=False)
            
            for i in idxs:
                a, b = neighbors[i]
                self.add_edge((r, c), (a, b))    

    def deposit_pheromone(self, pheromone: np.ndarray, row: int, col: int):
        self.pheromones[row, col] = pheromone
    
    def deposit_pheromone_delta(self, pheromone_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray], 
                                neighborhood_func: Callable[[float], float],
                                row: int, col: int, neighborhood: bool = True):
        centroid, node = self.get_centroid_pheromone_vec(row, col), self.get_pheromone_vec(row, col)
        # TODO: figure out the scale to use
        self.pheromones[row, col] = pheromone_func(centroid, node, neighborhood_func(0))
        if neighborhood:
            neighbors = self.get_neighborhood(row, col, self.centroid_radius, [(row, col)])
            for r, c in neighbors:
                centroid = self.get_centroid_pheromone_vec(r, c, [(row, col)])
                node = self.get_pheromone_vec(r, c)
                # dist = max(abs(row - r), abs(col - c))
                dist = float(np.sqrt((row - r) ** 2 + (col - c) ** 2))
                self.pheromones[r, c] = pheromone_func(centroid, node, neighborhood_func(dist))

    def deposit_pheromone_droplet(self, pheromone_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray], 
                                  match_func: Callable[[np.ndarray], float], 
                                  row: int, col: int, neighborhood: bool = True, num_select: int = 4):
        centroid, node = self.get_centroid_pheromone_vec(row, col), self.get_pheromone_vec(row, col)
        # TODO: figure out the scale to use
        self.pheromones[row, col] = pheromone_func(centroid, node, 1)
        if neighborhood:
            neighbors = self.get_neighborhood(row, col, self.centroid_radius, [(row, col)])
            pheromones = [self.get_pheromone_vec(r, c) for r, c in neighbors]
            centroids = [self.get_centroid_pheromone_vec(r, c, [(r, c)]) for r, c in neighbors]
            matches = [match_func(centroids[i]) for i in range(len(pheromones))]

            idxs = np.argpartition(matches, -num_select)[-num_select:]
            for k in idxs:
                r, c = neighbors[k]
                second_neighbors = self.get_neighborhood(r, c, self.centroid_radius, [(r, c), (row, col)])
                second_pheromones = [self.get_pheromone_vec(r2, c2) for r2, c2 in second_neighbors]
                second_centroids = [self.get_centroid_pheromone_vec(r2, c2, [(r2, c2)]) for r2, c2 in neighbors]
                second_matches = [match_func(second_centroids[i]) for i in range(len(second_pheromones))]

                k2 = np.argmax(second_matches)
                r2, c2 = neighbors[k2]

                self.pheromones[r, c] = pheromone_func(centroids[k], pheromones[k], 1)
                self.pheromones[r2, c2] = pheromone_func(second_centroids[k2], second_pheromones[k2], 1)

    def evaporate_pheromones(self):
        self.pheromones = (self.evap_factor * self.pheromones) + ((1 - self.evap_factor) * self.init_pheromones)
    
    def evolve_pheromones(self, survival_pct: float = 0.3):
        norms = np.linalg.norm(self.pheromones, axis=-1)
        flat_norms = norms.flatten()
        num_killed = int((1 - survival_pct) * len(flat_norms))
        killed_idx = np.argpartition(flat_norms, num_killed - 1)[:num_killed]
        rows, cols = np.unravel_index(killed_idx, norms.shape)

        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            self.pheromones[r, c] = self.get_centroid_pheromone_vec(r, c, exclude_list=[(r, c)])
            self.pheromones[r, c] = self.pheromones[r, c] / np.linalg.norm(self.pheromones[r, c])

    def trim_neighbors(self, row: int, col: int, dist_func: Callable = inv_cos_dist):
        neighbors = self.get_neighborhood(row, col, self.centroid_radius, [(row, col)])
        vec = self.get_pheromone_vec(row, col)
        if len(neighbors) != 0:
            vecs = [self.get_pheromone_vec(r, c) for r, c in neighbors]
            dists = [dist_func(v, vec) for v in vecs]
            i = np.argmax(dists)
            self.remove_edge((row, col), neighbors[i])

    def erode_network(self, dist_func: Callable = inv_cos_dist, min_dist: float = 1):
        lens = np.array([[len(c) for c in r] for r in self.documents]) 
        idxr, idxc = lens.nonzero()
        for r, c in zip(idxr, idxc):
            doc_list = self.documents[r, c]
            vec_list = self.doc_vecs[r, c]
            node_vec = self.get_pheromone_vec(r, c)

            dists = [dist_func(v, node_vec) for v in vec_list]
            self.documents[r, c] = [doc for i, doc in enumerate(doc_list) if dists[i] < min_dist]
            self.doc_vecs[r, c] = [vec for i, vec in enumerate(vec_list) if dists[i] < min_dist]

    def deposit_document(self, row: int, col: int, document: str, vec: np.ndarray):
        self.documents[row, col] += [document]
        self.doc_vecs[row, col] += [vec]

    def build_count_heatmap(self, k: float = 1.117):
        w = self.documents.shape[0]
        length_func = np.vectorize(lambda x: len(x))

        len_map = length_func(self.documents)
        r, c = len_map.nonzero()
        doc_lens = length_func(self.documents[r, c])
        max_len = np.max(doc_lens)

        update_r, update_c = (len_map != -1).nonzero()

        if max_len > 0 and len(update_r) > 0:
            m = zeta(k)
            self.count_heatmap = np.zeros((w, w))

            heats = np.zeros_like(update_r, dtype=np.float64)
            for i, l in enumerate(doc_lens):
                dr, dc = np.abs(update_r - r[i]), np.abs(update_c - c[i])
                dr = np.where(dr > (w / 2), w - dr, dr)
                dc = np.where(dc > (w / 2), w - dc, dc)
                dists = np.maximum(np.maximum(dr, dc), 1)

                heats += l / (m * max_len * (dists ** k))

            self.count_heatmap[update_r, update_c] = heats
                
    def get_documents(self, row: int, col: int, radius: int = 0) -> List[str]:
        neighborhood = self.get_neighborhood(row, col, radius)
        docs = []
        for r, c in neighborhood:
            docs += self.documents[r, c]
        return docs

    def to_pickle(self, out_path: str):
        with open(out_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(in_path: str):
        with open(in_path, 'rb') as f:
            d = pickle.load(f)
        if not isinstance(d, LatticeNetwork):
            raise ValueError(f"unrecognized datatype: {type(d)}")
        return d

class SmallWorldNetwork(LatticeNetwork):
    def __init__(self, network_shape: Tuple, embedding_dimension: int, evap_factor: float, rewire_prob: float = 0.01,
                 centroid_radius: int = 1, rng: Optional[Generator] = None, zeros: bool = False):
        super().__init__(network_shape, embedding_dimension, evap_factor, centroid_radius, rng, zeros)
        self.rewire_network(rewire_prob)

    def rewire_network(self, p: float = 0.01):
        cell_shape = tuple(n - 1 for n in self.neighbors.shape)
        num_edges = 6 * reduce(lambda a, b: a * b, cell_shape)
        num_rewire = int(p * num_edges)

        rewire_pos = self.rng.choice(self.neighbors.shape[0], (num_rewire, 2))
        for np_p in rewire_pos:
            p = tuple(np_p)
            while True:
                q = tuple(self.rng.choice(self.neighbors.shape[0], 2))
                if q != p:
                    break
            a, b = p
            i = self.rng.choice(len(self.neighbors[a, b]))
            r = self.neighbors[a, b][i]

            super().remove_edge(p, r)
            super().add_edge(p, q)

class HierarchicalLattice:
    def __init__(self, num_levels: int, scale: int, top_width: int, embedding_dimension: int, evaporation_factor: int, 
                 centroid_radius: int = 1, zeros: bool = False, rng: Optional[Generator] = None):
        if scale <= 1:
            raise ValueError(f"found invalid scale {scale}, scale must be >= 1")
        if scale % 2 != 1:
            raise RuntimeWarning(f"hierarchical network performs better with odd scaling factor, found scaling of {scale}")

        self.scale = scale
        self.num_levels = num_levels
        self.widths = [top_width * (scale ** k) for k in range(num_levels)]
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random
        self.levels = [LatticeNetwork((w, w), embedding_dimension, evaporation_factor, centroid_radius, self.rng, zeros) for w in self.widths]

    def get_next_level_pos(self, r: int, c: int, l: int) -> Optional[Tuple[int, int]]:
        if l >= len(self.levels):
            return None
        offset = self.scale // 2
        return ((self.scale * r) + offset, (self.scale * c) + offset)

    def deposit_pheromone(self, pheromone: np.ndarray, row: int, col: int, level: int):
        self.levels[level].deposit_pheromone(pheromone, row, col)
    
    def deposit_pheromone_delta(self, pheromone_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray], 
                                neighborhood_func: Callable[[float], float],
                                row: int, col: int, level: int, neighborhood: bool = True):
        self.levels[level].deposit_pheromone_delta(pheromone_func, neighborhood_func, row, col, neighborhood)
        
    def evaporate_pheromones(self):
        for i in range(len(self.levels)):
            self.levels[i].evaporate_pheromones()
    
    def deposit_document(self, row: int, col: int, document: str, vec: np.ndarray):
        self.levels[-1].deposit_document(row, col, document, vec)

    def erode_network(self, dist_func: Callable = inv_cos_dist, min_dist: float = 1):
        self.levels[-1].erode_network(dist_func, min_dist=min_dist)
            
    def get_neighbors(self, level: int, row: Union[int, np.ndarray], col: Union[int, np.ndarray]) -> Union[List[Tuple], np.ndarray]:
        return self.levels[level].neighbors[row, col]
    
    def get_level_network(self, level: int) -> LatticeNetwork:
        return self.levels[level]
        
    def to_pickle(self, out_path: str):
        with open(out_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(in_path: str):
        with open(in_path, 'rb') as f:
            d = pickle.load(f)
        if not isinstance(d, HierarchicalLattice):
            raise ValueError(f"unrecognized datatype: {type(d)}")
        return d
