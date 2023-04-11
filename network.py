import numpy as np

from numpy.random import Generator
from scipy.stats import norm

from functools import reduce
from typing import List, Tuple, Union, Optional, Callable

class LatticeNetwork():
    def __init__(self, network_shape: Tuple, embedding_dimension: int, evap_factor: float, 
                 centroid_radius: int = 1, rng: Optional[Generator] = None, zeros: bool = False):
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

        # initialize 8-neighbor neighborhood adjacency list
        self.neighbors = np.ndarray(network_shape, dtype=list)
        for row in range(network_shape[0]):
            for col in range(network_shape[1]):
                self.neighbors[row][col] = []
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if a == 0 and b == 0:
                            continue 
                        self.neighbors[row, col] += [((row + a) % self.neighbors.shape[0], (col + b) % self.neighbors.shape[1])]
                # self.neighbors[row]

                # if row != 0:
                #     north = (row - 1, col)
                #     self.neighbors[row][col].append(north)

                #     if col != 0:
                #         northwest = (row - 1, col - 1)
                #         self.neighbors[row][col].append(northwest)

                #     if col != network_shape[1] - 1:
                #         northeast = (row - 1, col + 1)
                #         self.neighbors[row][col].append(northeast)
                
                # if row != network_shape[0] - 1:
                #     south = (row + 1, col)
                #     self.neighbors[row][col].append(south)

                #     if col != 0:
                #         southwest = (row + 1, col - 1)
                #         self.neighbors[row][col].append(southwest)

                #     if col != network_shape[1] - 1:
                #         southeast = (row + 1, col + 1)
                #         self.neighbors[row][col].append(southeast)

                # if col != 0:
                #     west = (row, col - 1)
                #     self.neighbors[row][col].append(west)

                # if col != network_shape[1] - 1:
                #     east = (row, col + 1)
                #     self.neighbors[row][col].append(east)
    
    def get_pheromone_vec(self, row: Union[int, np.ndarray], col: Union[int, np.ndarray]) -> np.ndarray:
        return self.pheromones[row, col]

    # TODO: do we want to enforce that the mean pheremone vectors are normalized?
    # tentative argument for not normalizing, bc the norm of a vector is indicative of the "strength" of a given topic
    # (eg. a "distracted" node will have a pheremone vector with a lower norm, which means even if it finds a match it's less likely to pick it)
    # This will result in distracted nodes being less likely to be traversed, with specialized nodes being picked instead
    def get_centroid_pheromone_vec(self, row: int, col: int, exclude_list : List[Tuple] = []) -> np.ndarray:
        region_points = np.array(list(self.get_neighborhood(row, col, self.centroid_radius, exclude_list=exclude_list)))
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

    def deposit_pheromone(self, pheromone: np.ndarray, row: int, col: int):
        self.pheromones[row, col] = pheromone
    
    def deposit_pheromone_delta(self, pheromone_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray], 
                                neighborhood_func: Callable[[float], float],
                                row: int, col: int):
        neighbors = self.get_neighborhood(row, col, self.centroid_radius, [(row, col)])
        centroid, node = self.get_centroid_pheromone_vec(row, col, [(row, col)]), self.get_pheromone_vec(row, col)
        # TODO: figure out the scale to use
        self.pheromones[row, col] = pheromone_func(centroid, node, neighborhood_func(0))
        for r, c in neighbors:
            centroid = self.get_centroid_pheromone_vec(r, c, [(row, col)])
            node = self.get_pheromone_vec(r, c)
            dist = float(np.sqrt((row - r) ** 2 + (col - c) ** 2))
            self.pheromones[r, c] = pheromone_func(centroid, node, neighborhood_func(dist))

    def evaporate_pheromones(self):
        self.pheromones = (self.evap_factor * self.pheromones) + ((1 - self.evap_factor) * self.init_pheromones)