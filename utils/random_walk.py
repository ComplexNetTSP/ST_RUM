# from torch_cluster import random_walk
import dgl
import torch
from torch_geometric.utils.undirected import is_undirected


#RW with DGL
def uniform_random_walk(edge_index, edge_weight,
                        nodes, batch_size, num_samples,
                        timesteps,length, bias_walk=True):
    num_nodes = len(nodes)
    #flip
    source, target = edge_index[1], edge_index[0]
    dgl_graph = dgl.graph((source, target))
    nodes = nodes.repeat(batch_size * num_samples * timesteps)

    if bias_walk:
        dgl_graph.edata["p"] = edge_weight
        walks, _ = dgl.sampling.random_walk(dgl_graph, nodes, length=length,
                                            return_eids=False, prob='p', restart_prob = 0.)
    else:
        walks, _ = dgl.sampling.random_walk(dgl_graph, nodes, length=length,
                                            return_eids=False, restart_prob = 0.)
    
    walks = walks.view(batch_size, num_samples, timesteps, num_nodes, length+1)
    # eids = eids.view(batch_size, num_samples, timesteps, num_nodes, length)
    # if directed:
    walks = torch.where(
        walks == -1,
        walks[..., 0:1],
        walks,
    )
    
    return walks


def uniqueness(walk):
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    # (1 * walk_equal) -- > bool to int such that can use argmax
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal