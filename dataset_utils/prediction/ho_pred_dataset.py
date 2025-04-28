from tsl.data import SpatioTemporalDataset
import numpy as np
import toponetx as tnx
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from topomodelx.utils.sparse import from_sparse
from torch_sparse import SparseTensor
from utils import *
import torch
import warnings
warnings.filterwarnings("ignore")

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
        # self._graph = None
        # self._triangles = None
        # self._sc = None        
        # Now compute matrices after all properties are properly set up
        self.sparse = sparse
        self.signed = signed
        self.B1, self.B1T, self.B2, self.B2T = self._compute_incident_matrices()
        self.L0, self.L1, self.L2 = self._compute_laplacian_matrices()
    
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
    def simplicial_complex(self):
        """Property that lazily creates and caches the simplicial complex."""
        # if self._sc is None:
        edge_set = np.array(self.edge_index.T)
        self._sc = tnx.SimplicialComplex(self.nodes + list(edge_set) + self.triangles)
        return self._sc
    
    def inter_order_rw_matrix(self):
        total_size = self.L0.shape[0] + self.L1.shape[0] + self.L2.shape[0]
        block_matrix = torch.zeros(total_size, total_size)
        
        pass
    
    def _compute_incident_matrices(self):
        """compute incident matrices."""
        sc = self.simplicial_complex  # Use the property

        B1_N, B1T_N = compute_B1_B1T_normalized_matrix(sc, sparse=self.sparse, signed=self.signed)
        B2_N, B2T_N = compute_B2_B2T_normalized_matrix(sc, sparse=self.sparse, signed=self.signed)

        if self.sparse:
            B1_N, B1T_N =SparseTensor.from_torch_sparse_coo_tensor(from_sparse(B1_N)), SparseTensor.from_torch_sparse_coo_tensor(from_sparse(B1T_N))
            B2_N, B2T_N =SparseTensor.from_torch_sparse_coo_tensor(from_sparse(B2_N)), SparseTensor.from_torch_sparse_coo_tensor(from_sparse(B2T_N))
        else:   
            B1_N, B1T_N = torch.tensor(B1_N), torch.tensor(B1T_N)
            B2_N, B2T_N = torch.tensor(B2_N), torch.tensor(B2T_N)

        
        # B1 = sc.incidence_matrix(1, signed=False)
        # B2 = sc.incidence_matrix(2, signed=False)
        return B1_N, B1T_N, B2_N, B2T_N
    
    def _compute_laplacian_matrices(self):
        """compute Laplacian matrices."""
        sc = self.simplicial_complex
        L0, L1, L2 = normalized_high_order_adjacency(sc, sparse=self.sparse, signed=self.signed)

        if self.sparse:
            L0, L1, L2 = SparseTensor.from_torch_sparse_coo_tensor(from_sparse(L0)), SparseTensor.from_torch_sparse_coo_tensor(from_sparse(L1)), SparseTensor.from_torch_sparse_coo_tensor(from_sparse(L2))
        else:   
            L0, L1, L2 = torch.tensor(L0), torch.tensor(L1), torch.tensor(L2)

        return L0, L1, L2

        # L0, L1, L2 = normalized_high_order_adjcency(self.B1.todense().A, self.B2.todense().A)
        # return (torch.Tensor(L0).to_sparse_coo(),
        #         torch.Tensor(L1).to_sparse_coo(),
        #         torch.Tensor(L2).to_sparse_coo())

    def get(self, item):
        sample = super().get(item)
        # sample.input['B1'] = from_sparse(self.B1)
        # sample.input['B1T'] = from_sparse(self.B1T)
        # sample.input['B2'] = from_sparse(self.B2)
        # sample.input['B2T'] = from_sparse(self.B2T)

        sample.input['B1'] = self.B1
        sample.input['B1T'] = self.B1T
        sample.input['B2'] = self.B2
        sample.input['B2T'] = self.B2T
        
        sample.input['L0'] = self.L0
        sample.input['L1'] = self.L1
        sample.input['L2'] = self.L2
        
        return sample