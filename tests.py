import numpy as np
import matplotlib.pyplot as plt
import argparse
from network import LatticeNetwork, inv_cos_dist
from ant import Ant
from model import load_model
from functools import reduce
from main import ant_search

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--width", type=int, default=100)
    args.add_argument("-n", "--num-ants", type=int, default=1000)
    args.add_argument("-a", "--alpha", type=float, default=1)
    args.add_argument("-b", "--beta", type=float, default=2.7)
    args.add_argument("-d", "--delta", type=float, default=2)
    args.add_argument("-k", "--reinforce-exp", type=float, default=3)
    args.add_argument("-i", "--input-data", type=str, default='test_embed.pkl')
    args.add_argument("-s", "--num-steps", type=int, default=200)
    args.add_argument("-v", "--evaporation-factor", type=float, default=0.995)
    args.add_argument("-r", "--centroid-radius", type=int, default=1)
    args.add_argument("-z", "--zeros", action='store_true')
    args.add_argument("-q", "--greedy-prob", type=float, default=0.2)
    args.add_argument("-m", "--warmup-steps", type=int, default=50)
    args.add_argument("-e", "--export-video", action='store_true')
    args.add_argument("-g", "--geometry", type=str, choices=["euclidean", "graph"], default="euclidean")
    return args.parse_args()

if __name__ == '__main__':
    rng = np.random.default_rng(seed=0)
    args = parse_args()

    # import network and its documents from pkl
    path = input("Input Network Path: ")
    acses_network = LatticeNetwork.from_pickle(path)
    all_docs = np.vstack(reduce(lambda a,b: a+b, acses_network.documents.flatten())).flatten()
    all_doc_vecs = reduce(lambda a,b: a+b, acses_network.doc_vecs.flatten())

    model = load_model()

    random_placements = np.random.randint(low=0, high=args.width, size=(all_docs.shape[0], 2))

    # taken from https://stackoverflow.com/questions/43483663/how-do-i-make-a-grid-of-empty-lists-in-numpy-that-accepts-appending#:~:text=m%20%3D%20np.empty((12%2C%2012)%2C%20dtype%3Dobject)%0Afor%20i%20in%20np.ndindex(m.shape)%3A%20m%5Bi%5D%20%3D%20%5B%5D
    basic_network_docs = np.empty((args.width, args.width), dtype=object)
    for i in np.ndindex(basic_network_docs.shape): basic_network_docs[i] = []

    basic_network_vecs = np.ndarray((args.width,args.width), dtype=list)
    for i in np.ndindex(basic_network_vecs.shape): basic_network_vecs[i] = []

    for i in range(all_docs.shape[0]):
        basic_network_docs[random_placements[i,0]][random_placements[i,1]].append(all_docs[i])
        basic_network_vecs[random_placements[i,0]][random_placements[i,1]].append(all_doc_vecs[i])

    # basic_network_docs = acses_network.documents
    # basic_network_vecs = acses_network.doc_vecs


    doc_idxs = []

    # iterate through N sample documents,
    # calculate the inv_cos_dist from all docs at a randomly chosen node for the doc
    # calculate the inv_cod_dist from all docs at a ACSeS chosen node for the doc
    baseline_means = []
    baseline_mins = []
    baseline_maxs = []

    acses_means = []
    acses_mins = []
    acses_maxs = []
    acses_matches = []

    path_map = np.zeros((args.width, args.width))

    idxs = np.arange(all_docs.shape[0])
    np.random.shuffle(idxs)
    # sample N documents and compare ACSeS vs random allocation 
    # while len(doc_idxs) < args.num_ants:
    for doc_idx in idxs:
        # get a random document index
        # doc_idx = np.random.randint(low=0, high=all_docs.shape[0])
        doc_idxs += [doc_idx]

        # using ACSeS
        # initialize ant
        ant_vec = all_doc_vecs[doc_idx]
        loc = tuple(rng.choice(np.arange(args.width), 2))
        ant = Ant(ant_vec, loc, args.alpha, args.beta, args.delta, reinforce_exp=args.reinforce_exp, ant_id=0, document=all_docs[doc_idx], geometry=args.geometry)

        data = ant_search(acses_network, ant, q=args.greedy_prob, max_steps=200)
        if data is not None:
            ant, acses_docs, pos_seq, pher_seq = data
            for r, c in pos_seq:
                path_map[r, c] += 1
            # get the vectors stored in the ACSeS-selected node
            acses_node_idx = pos_seq[-1]
            acses_vecs = acses_network.doc_vecs[acses_node_idx[0]][acses_node_idx[1]]

            # calculate the inverse cosine distance of the current doc with all docs in the randomly selected node
            acses_dists = [inv_cos_dist(all_doc_vecs[doc_idx], vec) for vec in acses_vecs]
            acses_matches += [ant.current_pheromone]
            l = len(pos_seq)
        else:
            acses_dists = [2.0]
            l = 200

        # store acses stats
        acses_means += [np.mean(acses_dists)]
        acses_mins += [np.min(acses_dists)]
        acses_maxs += [np.max(acses_dists)]

        # baseline inference
        tried_set = set()
        while len(tried_set) < l:
            baseline_node_idx = np.random.randint(low=0, high=args.width, size=2)
            if tuple(baseline_node_idx) in tried_set:
                continue
            tried_set.add(tuple(baseline_node_idx))

            # get the document and vectors stored in the randomly selected node
            baseline_docs = basic_network_docs[baseline_node_idx[0], baseline_node_idx[1]]
            baseline_vecs = basic_network_vecs[baseline_node_idx[0], baseline_node_idx[1]]

            # calculate the inverse cosine distance of the current doc with all docs in the randomly selected node
            baseline_dists = [inv_cos_dist(all_doc_vecs[doc_idx], vec) for vec in baseline_vecs]
            if len(baseline_dists) != 0:
                break
            # baseline_dists = [2.0]
        if len(baseline_dists) == 0:
            baseline_dists = [2.0]

        # store baseline stats
        baseline_means += [np.mean(baseline_dists)]
        baseline_mins += [np.min(baseline_dists)]
        baseline_maxs += [np.max(baseline_dists)]


    plt.imshow(path_map)
    plt.title("Ant Path Trails")
    plt.show()

    acses_eff = len([a for a in acses_mins if a == 2.0]) / len(acses_mins)
    baseline_eff = len([b for b in baseline_mins if b == 2.0]) / len(baseline_mins)

    print(f"ACSeS Search Failure Rate: {acses_eff}")
    print(f"Baseline Search Failure Rate: {baseline_eff}")
   
    f,axs = plt.subplots(3, sharey=True)

    b_idx = [i for i, b in enumerate(baseline_mins) if b != 2.0]

    b_mins = np.array(baseline_mins)
    b_means = np.array(baseline_means)
    b_maxs = np.array(baseline_maxs)

    print(f"Average Baseline Upper-Bound: {np.mean(b_maxs[b_idx])}")
    print(f"Average Baseline Lower-Bound: {np.mean(b_mins[b_idx])}")

    axs[0].scatter(b_idx, b_mins[b_idx], label='Min Distance: Baseline')
    axs[1].scatter(b_idx, b_maxs[b_idx], label='Max Distance: Baseline')
    axs[2].scatter(b_idx, b_means[b_idx], label='Mean Distance: Baseline')

    a_idx = [i for i, a in enumerate(acses_mins) if a != 2.0]

    a_mins = np.array(acses_mins)
    a_means = np.array(acses_means)
    a_maxs = np.array(acses_maxs)

    print(f"Average ACSeS Upper-Bound: {np.mean(a_maxs[a_idx])}")
    print(f"Average ACSeS Lower-Bound: {np.mean(a_mins[a_idx])}")

    axs[0].scatter(a_idx, a_mins[a_idx], label='Min Distance: ACSeS')
    axs[1].scatter(a_idx, a_maxs[a_idx], label='Max Distance: ACSeS')
    axs[2].scatter(a_idx, a_means[a_idx], label='Mean Distance: ACSeS')

    axs[0].legend()
    axs[0].set_xticks([])
    # axs[0].set_title('Minimum Inverse Cosine Distance between Query and Retrieved Documents')
    axs[1].legend()
    axs[1].set_xticks([])
    axs[1].set_ylabel('Inverse Cosine Distance')
    # axs[1].set_title('Maximum Inverse Cosine Distance between Query and Retrieved Documents')
    axs[2].legend()
    axs[2].set_xticks([])
    # axs[2].set_title('Mean Inverse Cosine Distance between Query and Retrieved Documents')

    plt.suptitle('Inverse Cosine Distance Between Query and Retrieved Documents')

    plt.show()

    
    
