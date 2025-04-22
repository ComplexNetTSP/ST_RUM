import torch
from torch_cluster import random_walk


def uniform_random_walk(edge_index, nodes, num_samples, length):
    """
    Random walk on a graph.

    Parameters
    ----------
    edge_index : torch
    num_samples : int
        Number of random walks per node.
    length : int
        Length of each random walk.

    Returns
    -------
    walks : Tensor
        The random walks.
    """
    source, target = edge_index[0], edge_index[1]
    
    num_nodes = len(nodes)
    nodes = nodes.repeat(num_samples)

    walks, eids = random_walk(row=source,
                              col=target,
                              start=nodes,
                              walk_length=length,
                              return_edge_indices = True)
    
    walks = walks.view(num_samples, num_nodes, length+1)
    eids = eids.view(num_samples, num_nodes, length)
    
    return walks, eids


def uniqueness(walk):
    """
    Compute the uniqueness of a random walk.

    Parameters
    ----------
    walk : Tensor
        The random walk.

    Returns
    -------
    uniqueness : Tensor
        The uniqueness of the random walk.
    """
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    # (1 * walk_equal) -- > bool to int such that can use argmax
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal

