from torch_cluster import random_walk

def uniform_random_walk(edge_index, nodes, batch_size, num_samples,timesteps,length):
    source, target = edge_index[0], edge_index[1]
    
    num_nodes = len(nodes)
    nodes = nodes.repeat(batch_size * num_samples * timesteps)

    walks, eids = random_walk(row=source,
                              col=target,
                              start=nodes,
                              walk_length=length,
                              return_edge_indices = True)
    
    walks = walks.view(batch_size, num_samples, timesteps, num_nodes, length+1)
    eids = eids.view(batch_size, num_samples, timesteps, num_nodes, length)
    
    return walks, eids


def uniqueness(walk):
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    # (1 * walk_equal) -- > bool to int such that can use argmax
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal