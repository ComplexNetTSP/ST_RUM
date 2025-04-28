from tsl.data import SpatioTemporalDataset
import numpy as np
import toponetx as tnx
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_sparse import SparseTensor
from utils import *
import itertools
from torch_geometric.utils.sparse import to_edge_index
import torch
import warnings
warnings.filterwarnings("ignore")

def find_cliques_size_k(G, k):
    all_cliques = set()
    for clique in nx.find_cliques(G):
        if len(clique) == k:
            all_cliques.add(tuple(sorted(clique)))
        elif len(clique) > k:
            for mini_clique in itertools.combinations(clique, k):
                all_cliques.add(tuple(sorted(mini_clique)))
    return list(all_cliques)

class HO_Pre(SpatioTemporalDataset):
    def __init__(self, 
                 target,
                 mask=None,
                 connectivity=None,
                 covariates=None,
                 scalers=None,
                 window=24,
                 horizon=2,
                 stride=1,
                 sparse = True,
                 signed = True,
                 *args, **kwargs):
        
        super().__init__(target = target,
                        mask=mask,
                        connectivity=connectivity,
                        covariates=covariates,
                        scalers=scalers,
                        window=window,
                        horizon=horizon,
                        stride=stride,
                        *args, **kwargs)     
        # Now compute matrices after all properties are properly set up
        self.sparse = sparse
        self.signed = signed
        self.rw_edge_index = self.inter_order_rw_matrix()
    
    @property
    def graph(self):
        """Property that lazily creates and caches the graph."""
        # if self._graph is None:
        # tensor_graph = to_dense_adj(self.edge_index, max_num_nodes=self.n_nodes).squeeze(0)
        sparse_graph = to_scipy_sparse_matrix(self.edge_index, self.edge_weight, num_nodes=self.n_nodes)
        # array_graph = np.array(tensor_graph)
        # self._graph = nx.from_numpy_array(array_graph)
        self._graph = nx.from_scipy_sparse_array(sparse_graph)
        return self._graph
    
    @property
    def nodes(self):
        """Property that lazily creates and caches the nodes."""
        return list(self.graph.nodes)
        
    @property
    def triangles(self):
        """Property that lazily computes and caches the triangles."""
        # if self._triangles is None:
        self._triangles = find_cliques_size_k(self.graph, 3)
        return self._triangles
    
    @property
    def simplical_complex(self):
        """Property that lazily creates and caches the simplicial complex."""
        # if self._sc is None:
        edge_set = np.array(self.edge_index.T)
        self._sc = tnx.SimplicialComplex(self.nodes + list(edge_set) + self.triangles)
        return self._sc

    def create_edge_dict(self, edge_index, edge_weight):
        return {(edge_index[0, i].item(), edge_index[1, i].item()): 
                {"edge_feature": edge_weight[i].item()} 
                for i in range(edge_index.shape[1])}
    
    def inter_order_rw_matrix(self):
        simplical_complex = self.simplical_complex
        simplical_complex.set_simplex_attributes(self.create_edge_dict(self.edge_index, self.edge_weight))
        
        new_edge_feature = torch.Tensor(np.array(list(simplical_complex.get_simplex_attributes("edge_feature").values())))
        
        L0_up = simplical_complex.adjacency_matrix(rank=0, index=False, signed = False).todense().A
        L1_up = simplical_complex.adjacency_matrix(rank=1, index=False, signed = False).todense().A
        L1_down = simplical_complex.coadjacency_matrix(rank=1, index=False, signed = False).todense().A
        L1 = L1_up + L1_down
        L2_down = simplical_complex.coadjacency_matrix(rank=2, index=False, signed = False).todense().A

        L0_up = torch.tensor(L0_up, dtype=torch.float) if isinstance(L0_up, np.ndarray) else L0_up
        L1 = torch.tensor(L1, dtype=torch.float) if isinstance(L1, np.ndarray) else L1
        L2_down = torch.tensor(L2_down, dtype=torch.float) if isinstance(L2_down, np.ndarray) else L2_down

        B1 = simplical_complex.incidence_matrix(rank=1, index=False, signed = False).todense().A # nodes x edges
        B1 = torch.tensor(B1, dtype=torch.float) if isinstance(B1, np.ndarray) else B1
        B1T = B1.T
        
        B2 = simplical_complex.incidence_matrix(rank=2, index=False, signed = False).todense().A # edges x triangles
        B2 = torch.tensor(B2, dtype=torch.float) if isinstance(B2, np.ndarray) else B2
        B2T = B2.T

        new_face_feature = B2T @ new_edge_feature

        
        total_size = L0_up.shape[0] + L1.shape[0] + L2_down.shape[0]
        block_matrix = torch.zeros(total_size, total_size)

        # Fill the diagonal blocks
        block_matrix[:L0_up.shape[0], :L0_up.shape[1]] = L0_up
        block_matrix[L0_up.shape[0]:L0_up.shape[0]+L1.shape[0], L0_up.shape[1]:L0_up.shape[1]+L1.shape[1]] = L1
        block_matrix[L0_up.shape[0]+L1.shape[0]:, L0_up.shape[1]+L1.shape[1]:] = L2_down

        # Fill the upper off-diagonal blocks
        block_matrix[:L0_up.shape[0], L0_up.shape[1]:L0_up.shape[1]+L1.shape[0]] = B1
        block_matrix[L0_up.shape[0]:L0_up.shape[0]+L1.shape[0], L0_up.shape[1]+L1.shape[1]:] = B2

        # Fill the lower off-diagonal blocks
        block_matrix[L0_up.shape[0]:L0_up.shape[0]+L1.shape[0], :L0_up.shape[1]] = B1T
        block_matrix[L0_up.shape[0]+L1.shape[0]:, L0_up.shape[1]:L0_up.shape[1]+L1.shape[1]] = B2T
        

        sparse_block_matrix = SparseTensor.from_dense(block_matrix)

        rw_edge_index = to_edge_index(sparse_block_matrix)    

        return rw_edge_index[0], new_edge_feature, new_face_feature

    def get(self, item):
        sample = super().get(item)

        rw_edge_index, edge_feature, triangle_feature = self.rw_edge_index
        
        sample.input['rw_edge_index'] = rw_edge_index
        sample.input['edge_feature'] = edge_feature
        sample.input['triangle_feature'] = triangle_feature
        
        return sample