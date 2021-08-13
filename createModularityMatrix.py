import numpy as np
import networkx as nx
from createRatingMatrix import create_ratings


def create_mMatrix(trainset, tpath):
    G = nx.DiGraph()
    G.add_nodes_from(list(trainset.all_users()))
    with open(tpath, 'r') as file:
        for line in file.readlines():
            user1 = line.split(' ')[0]
            user2 = line.split(' ')[1]
            try:
                inner_u1 = trainset.to_inner_uid(user1)
                inner_u2 = trainset.to_inner_uid(user2)
                G.add_edge(inner_u1, inner_u2)
            except ValueError:
                pass
    return np.asarray(nx.directed_modularity_matrix(G)), np.asarray(nx.adjacency_matrix(G).todense())


if __name__ == '__main__':
    rpath = "C:/gjw2/Rec/Datasets/filmtrust/cvtrain1.txt"
    rating_scale = (0.5, 4)
    trainset = create_ratings(rpath, rating_scale)
    tpath = "C:/gjw2/Rec/Datasets/filmtrust/trust.txt"
    print(create_mMatrix(trainset, tpath))
