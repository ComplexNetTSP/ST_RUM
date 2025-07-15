from tsl.data import ImputationDataset
import numpy as np
import pandas as pd
import toponetx as tnx
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_sparse import SparseTensor
from utils import *
import itertools
from torch_geometric.utils.sparse import to_edge_index
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
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


def prepare_high_order_feature(points, coord_type='relative', mode='delaunay', edge_index=None, triangles=None):
    """
    Prepare high order features including edges and triangles
    
    Args:
        points: coordinates array (N, 2) or DataFrame
        coord_type: 'relative' for x,y coordinates or 'geographic' for lat,lon
        mode: 'delaunay' or 'graph'
        edge_index: for graph mode, array of shape (2, num_edges) or list of tuples
        triangles: for graph mode, array of shape (num_triangles, 3) or list of tuples
    
    Returns:
        edge_dict: {(node1, node2): {"edge_features": distance}}
        triangle_dict: {(node1, node2, node3): {"triangle_features": area}}
    """
    
    # Convert points to numpy array if it's a DataFrame
    if isinstance(points, pd.DataFrame):
        points_array = points.values
    else:
        points_array = points
    
    # Calculate distance matrix based on coordinate type
    if coord_type == 'geographic':
        # For lat,lon coordinates
        from tsl.ops.similarities import geographical_distance
        distance_matrix = geographical_distance(points, to_rad=True)
        if isinstance(distance_matrix, pd.DataFrame):
            distance_matrix = distance_matrix.values
    else:
        # For relative x,y coordinates
        distance_matrix = squareform(pdist(points_array))
    
    if mode == 'delaunay':
        # Create Delaunay triangulation
        tri = Delaunay(points_array)
        
        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([int(simplex[i]), int(simplex[j])]))
                    edges.add(edge)
        
        # Get triangles from Delaunay
        triangle_list = [tuple([int(x) for x in simplex]) for simplex in tri.simplices]
        
    elif mode == 'graph':
        # Use provided edge_index and triangles
        if edge_index is None or triangles is None:
            raise ValueError("For graph mode, both edge_index and triangles must be provided")
        
        # Convert edge_index to set of tuples
        if isinstance(edge_index, torch.Tensor) and edge_index.shape[0] == 2:
            # Shape (2, num_edges)
            edges = set()
            for i in range(edge_index.shape[1]):
                edge = tuple(sorted([int(edge_index[0, i]), int(edge_index[1, i])]))
                edges.add(edge)
        else:
            # List of tuples or other format
            edges = set()
            for edge in edge_index:
                edges.add(tuple(sorted([int(edge[0]), int(edge[1])])))
        
        # Convert triangles to list
        if isinstance(triangles, np.ndarray):
            triangle_list = [tuple([int(x) for x in triangle]) for triangle in triangles]
        else:
            triangle_list = [tuple([int(x) for x in triangle]) for triangle in triangles]
    
    else:
        raise ValueError("mode must be 'delaunay' or 'graph'")
    
    # Create edge dictionary with distances
    edge_dict = {edge: {"edge_features": float(distance_matrix[edge[0], edge[1]])} for edge in edges}
    # Create triangle dictionary with areas
    triangle_dict = {}
    for triangle in triangle_list:
        if coord_type == 'geographic':
            # For geographic coordinates, use Heron's formula
            a = distance_matrix[triangle[0], triangle[1]]  # distance p1-p2
            b = distance_matrix[triangle[1], triangle[2]]  # distance p2-p3  
            c = distance_matrix[triangle[0], triangle[2]]  # distance p1-p3
            
            # Use Heron's formula for area
            s = (a + b + c) / 2  # semi-perimeter
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            
        else:
            # For relative coordinates, use cross product
            p1, p2, p3 = points_array[triangle[0]], points_array[triangle[1]], points_array[triangle[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            area = 0.5 * abs(np.cross(v1, v2))
        
        triangle_dict[triangle] = {"triangle_features": float(area)}
    
    return edge_dict, triangle_dict


class HO_Imp(ImputationDataset):
    """Higher-Order Imputation Dataset - extends ImputationDataset with higher-order adjacency matrices"""
    
    def __init__(self, 
                 target,
                 eval_mask,
                 mask=None,
                 connectivity=None,
                 covariates=None,
                 scalers=None,
                 window=24,
                 stride=1,
                 sparse=True,
                 signed=False,
                 order=2,  # 0: node-only, 1: node+edge, 2: node+edge+triangle
                 diagonal=True,  # Include diagonal blocks (L0, L1, L2) or only off-diagonal
                 bias=True,  # Bias or unbias random walk
                 norm='row',  # 'row', 'col', or None for normalization
                 # New hyperparameters for weighting
                 points=None,  # Coordinate points for Delaunay or geographic features
                 coord_type='relative',  # 'relative' or 'geographic'
                 use_delaunay=False,  # Use Delaunay triangulation instead of clique finding
                 *args, **kwargs):
        
        super().__init__(target=target,
                        eval_mask=eval_mask,
                        mask=mask,
                        connectivity=connectivity,
                        covariates=covariates,
                        scalers=scalers,
                        window=window,
                        stride=stride,
                        *args, **kwargs)
        
        self.sparse = sparse
        self.signed = signed
        self.order = order
        self.diagonal = diagonal
        self.bias = bias
        self.norm = norm
        self.points = points
        self.coord_type = coord_type
        self.use_delaunay = use_delaunay
        self.rw_matrices = self.inter_order_rw_matrix()
        # Print dataset statistics
        print(f"Dataset created: {self.n_nodes} nodes", end="")
        if self.order >= 1:
            num_edges = self.edge_index.shape[1] if hasattr(self, 'edge_index') else len(self.graph.edges)
            print(f", {num_edges} edges", end="")
        if self.order >= 2:
            print(f", {len(self.triangles)} triangles")
        else:
            print()  # New line
    
    @property
    def graph(self):
        sparse_graph = to_scipy_sparse_matrix(self.edge_index, self.edge_weight, num_nodes=self.n_nodes)
        self._graph = nx.from_scipy_sparse_array(sparse_graph)
        return self._graph
    
    @property
    def nodes(self):
        return list(self.graph.nodes)
        
    @property
    def triangles(self):
        if self.order < 2:
            return []
        
        if self.use_delaunay and self.points is not None:
            # Use Delaunay triangulation
            tri = Delaunay(self.points)
            self._triangles = [tuple([int(x) for x in simplex]) for simplex in tri.simplices]
        else:
            # Use clique finding from existing graph
            self._triangles = find_cliques_size_k(self.graph, 3)
            
        return self._triangles
    
    @property
    def simplical_complex(self):
        components = [self.nodes]
        if self.order >= 1:
            components.append(list(np.array(self.edge_index.T)))
        if self.order >= 2:
            components.append(self.triangles)

        self._sc = tnx.SimplicialComplex([item for sublist in components for item in sublist])
        return self._sc

    def create_feature_dicts(self):
        """Create edge and triangle feature dictionaries using prepare_high_order_feature"""
        if self.points is None:
            # Fallback to original method if no coordinates provided
            return {(self.edge_index[0, i].item(), self.edge_index[1, i].item()): 
                   {"edge_features": self.edge_weight[i].item()} 
                   for i in range(self.edge_index.shape[1])}, {}
        
        if self.use_delaunay:
            # Use Delaunay triangulation
            edge_dict, triangle_dict = prepare_high_order_feature(
                self.points, 
                coord_type=self.coord_type, 
                mode='delaunay'
            )
        else:
            # Use existing graph structure
            edge_dict, triangle_dict = prepare_high_order_feature(
                self.points, 
                coord_type=self.coord_type, 
                mode='graph',
                edge_index=self.edge_index,
                triangles=self.triangles
            )
        
        return edge_dict, triangle_dict
    
    def inter_order_rw_matrix(self):
        if self.order == 0:
            return {}  # Use existing edge_index and edge_weight
        elif self.order == 1:
            return self._build_order_1()
        else:  # order == 2
            return self._build_order_2()
    
    def _build_order_1(self):
        # Node + Edge level with weighting:
        
        
        sc = self.simplical_complex
        edge_dict, _ = self.create_feature_dicts()
        sc.set_simplex_attributes(edge_dict)
        
        edge_features = torch.tensor(np.array(list(sc.get_simplex_attributes("edge_features").values())))
        
        L0 = torch.tensor(sc.adjacency_matrix(rank=0, signed=self.signed).todense().A, dtype=torch.float)
        
        # For order=1, use coadjacency_matrix which handles edge-to-edge connections better
        try:
            L1 = torch.tensor(sc.coadjacency_matrix(rank=1, signed=self.signed).todense().A, dtype=torch.float)
        except ValueError:
            # If no edge-to-edge connections exist, create zero matrix
            num_edges = len(list(sc.skeleton(1))) - len(list(sc.skeleton(0)))  # edges only
            L1 = torch.zeros(num_edges, num_edges, dtype=torch.float)
        
        B1 = torch.tensor(sc.incidence_matrix(rank=1, signed=self.signed).todense().A, dtype=torch.float)
        
        # Build block matrix
        total_size = L0.shape[0] + L1.shape[0]
        block_matrix = torch.zeros(total_size, total_size)
        
        if self.bias:
            # Weighted blocks for biased walk
            if self.diagonal:
                # [[α*L0, (1-α)*B1],
                # [β*B1T, (1-β)*L1]]
                block_matrix[:L0.shape[0], :L0.shape[0]] = (1/L0.shape[0])/(1/L0.shape[0] + 1/L1.shape[0]) * L0
                block_matrix[L0.shape[0]:, L0.shape[0]:] = (1/L1.shape[0])/(1/L0.shape[0] + 1/L1.shape[0]) * L1
                block_matrix[:L0.shape[0], L0.shape[0]:] = (1/L1.shape[0])/(1/L0.shape[0] + 1/L1.shape[0]) * B1
                block_matrix[L0.shape[0]:, :L0.shape[0]] = (1/L0.shape[0])/(1/L0.shape[0] + 1/L1.shape[0]) * B1.T
            else:
                # [[0, B1],
                # [B1T, 0]]   
                block_matrix[:L0.shape[0], L0.shape[0]:] = B1
                block_matrix[L0.shape[0]:, :L0.shape[0]] = B1.T
        else:
            # Unweighted blocks for unbiased walk
            if self.diagonal:
                block_matrix[:L0.shape[0], :L0.shape[0]] = L0
                block_matrix[L0.shape[0]:, L0.shape[0]:] = L1
            
            block_matrix[:L0.shape[0], L0.shape[0]:] = B1
            block_matrix[L0.shape[0]:, :L0.shape[0]] = B1.T
        
        # Apply bias/unbias normalization
        if self.bias and self.norm:
            if self.norm == 'row':
                # Row normalization for biased random walk
                row_sums = block_matrix.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                block_matrix = block_matrix / row_sums
            elif self.norm == 'col':
                # Column normalization
                col_sums = block_matrix.sum(dim=0, keepdim=True)
                col_sums[col_sums == 0] = 1  # Avoid division by zero
                block_matrix = block_matrix / col_sums
        
        sparse_matrix = SparseTensor.from_dense(block_matrix)
        rw_edge_index, rw_edge_weight = to_edge_index(sparse_matrix)

        # scaler for edge features
        edge_scaler = StandardScaler()
        edge_features = torch.tensor(edge_scaler.fit_transform(edge_features.reshape(-1, 1)), dtype=torch.float).reshape(-1)
        return {
            'rw_edge_index': rw_edge_index,
            'rw_edge_weight': rw_edge_weight,
            'edge_features': edge_features,
            'triangle_features': None
        }
    
    def _build_order_2(self):
        # Node + Edge + Triangle level with weighting:
        # [[α*L0, (1-α)*B1, 0], 
        # [β*B1T, (1-β)/2*L1, (1-β)/2*B2], 
        # [0, 0.5*B2T, 0.5*L2]]
        
        sc = self.simplical_complex
        edge_dict, triangle_dict = self.create_feature_dicts()

        # Set attributes for edges
        sc.set_simplex_attributes(edge_dict)
        
        # Set attributes for triangles if using coordinate-based features
        if triangle_dict:
            triangle_attr_dict = {}
            for triangle, attr in triangle_dict.items():
                triangle_attr_dict[triangle] = attr["triangle_features"]
            sc.set_simplex_attributes(triangle_attr_dict, name="triangle_features")
        
        edge_features = torch.tensor(np.array(list(sc.get_simplex_attributes("edge_features").values())), dtype=torch.float)
        
        L0 = torch.tensor(sc.adjacency_matrix(rank=0, signed=self.signed).todense().A, dtype=torch.float)
        L1 = torch.tensor((sc.adjacency_matrix(rank=1, signed=self.signed) + 
                          sc.coadjacency_matrix(rank=1, signed=self.signed)).todense().A, dtype=torch.float)
        L2 = torch.tensor(sc.coadjacency_matrix(rank=2, signed=self.signed).todense().A, dtype=torch.float)
        
        B1 = torch.tensor(sc.incidence_matrix(rank=1, signed=self.signed).todense().A, dtype=torch.float)
        B2 = torch.tensor(sc.incidence_matrix(rank=2, signed=self.signed).todense().A, dtype=torch.float)
        
        # Use coordinate-based triangle features if available, otherwise compute from edges
        if triangle_dict:
            triangle_features = torch.tensor(list(triangle_attr_dict.values()), dtype=torch.float)
        else:
            triangle_features = B2.T @ edge_features
        
        # Build block matrix
        total_size = L0.shape[0] + L1.shape[0] + L2.shape[0]
        block_matrix = torch.zeros(total_size, total_size)
        
        n0, n1 = L0.shape[0], L1.shape[0]
        
        if self.bias:
            # [[α*L0, (1-α)*B1, 0], 
            # [β*B1T, (1-β)/2*L1, (1-β)/2*B2], 
            # [0, 0.5*B2T, 0.5*L2]]
            # Weighted blocks for biased walk            
            if self.diagonal:
                block_matrix[:n0, :n0] = (1/L0.shape[0])/(1/L0.shape[0] + 1/L1.shape[0]) * L0
                block_matrix[n0:n0+n1, n0:n0+n1] = (1/L1.shape[0])/(1/L0.shape[0] + 1/L1.shape[0] + 1/L2.shape[0]) * L1
                block_matrix[n0+n1:, n0+n1:] = (1/L2.shape[0])/(1/L1.shape[0] + 1/L2.shape[0]) * L2

                block_matrix[:n0, n0:n0+n1] = (1/L1.shape[0])/(1/L0.shape[0] + 1/L1.shape[0]) * B1
                block_matrix[n0:n0+n1, :n0] = (1/L0.shape[0])/(1/L0.shape[0] + 1/L1.shape[0] + 1/L2.shape[0]) * B1.T
                block_matrix[n0:n0+n1, n0+n1:] = (1/L2.shape[0])/(1/L0.shape[0] + 1/L1.shape[0] + 1/L2.shape[0]) * B2
                block_matrix[n0+n1:, n0:n0+n1] = (1/L1.shape[0])/(1/L1.shape[0] + 1/L2.shape[0]) * B2.T
            else:
                block_matrix[:n0, n0:n0+n1] = B1
                block_matrix[n0:n0+n1, :n0] = (1/L0.shape[0])/(1/L0.shape[0] + 1/L2.shape[0]) * B1.T
                block_matrix[n0:n0+n1, n0+n1:] = (1/L2.shape[0])/(1/L0.shape[0] + 1/L2.shape[0]) * B2
                block_matrix[n0+n1:, n0:n0+n1] = B2.T
        else:
            # Unweighted blocks for unbiased walk
            if self.diagonal:
                block_matrix[:n0, :n0] = L0
                block_matrix[n0:n0+n1, n0:n0+n1] = L1
                block_matrix[n0+n1:, n0+n1:] = L2
            
            block_matrix[:n0, n0:n0+n1] = B1
            block_matrix[n0:n0+n1, :n0] = B1.T
            block_matrix[n0:n0+n1, n0+n1:] = B2
            block_matrix[n0+n1:, n0:n0+n1] = B2.T
        
        # Apply bias/unbias normalization
        if self.bias and self.norm:
            if self.norm == 'row':
                # Row normalization for biased random walk
                row_sums = block_matrix.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                block_matrix = block_matrix / row_sums
            elif self.norm == 'col':
                # Column normalization
                col_sums = block_matrix.sum(dim=0, keepdim=True)
                col_sums[col_sums == 0] = 1  # Avoid division by zero
                block_matrix = block_matrix / col_sums
        
        sparse_matrix = SparseTensor.from_dense(block_matrix)
        rw_edge_index, rw_edge_weight = to_edge_index(sparse_matrix)

        # scaler for edge/triangle features
        edge_scaler = StandardScaler()
        triangle_scaler = StandardScaler()

        edge_features = torch.tensor(edge_scaler.fit_transform(edge_features.reshape(-1, 1)), dtype=torch.float).reshape(-1)
        triangle_features = torch.tensor(triangle_scaler.fit_transform(triangle_features.reshape(-1, 1)), dtype=torch.float).reshape(-1)
        
        return {
            'rw_edge_index': rw_edge_index,
            'rw_edge_weight': rw_edge_weight,
            'edge_features': edge_features,
            'triangle_features': triangle_features
        }

    def get(self, item):
        sample = super().get(item)

        if self.order > 0:  # Only add higher-order features when needed
            for key, value in self.rw_matrices.items():
                if value is not None:
                    sample.input[key] = value
        
        return sample