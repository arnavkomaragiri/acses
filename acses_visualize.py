import numpy as np
import matplotlib.pyplot as plt

from network import LatticeNetwork
from ant import Ant
from main import find_pheromone_map, init_ant, ant_search
from model import load_model

if __name__ == "__main__":
    network_path = input("Input Saved Network Path: ")
    network = LatticeNetwork.from_pickle(network_path)

    model = load_model()
    sent = input("Input Query: ")
    emb = model.encode(sent)

    ant = init_ant(network, emb, 1, 2.7, 2, doc=sent)

    pher_map = find_pheromone_map(ant, network.pheromones, emb)
    plt.imshow(pher_map)
    plt.title(f"Pheromone Match to Query: {sent}")
    plt.show()

    data = ant_search(network, ant, 0.2, max_steps=200)
    if data is not None:
        a, docs, pos_seq, pher_seq = data
        print("Search Results:")
        for d in docs:
            print(d)
        path_map = np.zeros((network.pheromones.shape[0], network.pheromones.shape[1]), dtype=float)
        for i, (a, b) in enumerate(pos_seq):
            path_map[a, b] = (pher_seq[i] / np.max(pher_seq))
        print(f"Ant Walk Length: {len(pos_seq)}")
        plt.imshow(path_map)
        plt.title("Ant Path Colored by Pheromone Match")
        plt.show()
    else:
        print("Search Failed")