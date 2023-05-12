import numpy as np
import matplotlib.pyplot as plt

from network import HierarchicalLattice
from ant import HierarchicalAnt
from hierarchy_main import find_pheromone_map, init_ant, ant_search
from model import load_model

if __name__ == "__main__":
    network_path = input("Input Saved Network Path: ")
    network = HierarchicalLattice.from_pickle(network_path)

    model = load_model()
    sent = input("Input Query: ")
    emb = model.encode(sent)

    ant = init_ant(network, emb, 1, 32, 0.2, doc=sent)

    pher_map = find_pheromone_map(ant, network.levels[-1].pheromones, emb)
    plt.imshow(pher_map)
    plt.title(f"Pheromone Match to Query: {sent}")
    plt.show()

    data = ant_search(network, ant, 0.2, max_steps=200)
    if data is not None:
        a, docs, pos_seq, pher_seq = data
        print("Search Results:")
        for d in docs:
            print(d)
        print(f"Ant Walk Length: {len(pos_seq)}")
    else:
        print("Search Failed")