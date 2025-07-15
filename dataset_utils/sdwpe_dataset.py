import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
from tsl.datasets.prototypes import DatetimeDataset
from tsl.ops.similarities import gaussian_kernel

class SDWPE(DatetimeDataset):
    similarity_options = {'distance', 'precomputed'}
    
    def __init__(self, freq=None, impute_zeros=True):
        df, dist, mask, triangles = self.load(impute_zeros=impute_zeros)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         name="SDWPE")
        
        self.add_covariate('dist', dist, pattern='n n')
        self.add_covariate('denaulay_tri', triangles)
        
    def load_raw(self):

        location = pd.read_csv('data/sdwpe/sdwpf_turb_location_elevation.csv')
        dataset = pd.read_csv('data/sdwpe/sdwpf_2001_2112_full.csv', engine='pyarrow', parse_dates=['Tmstamp'])
        dataset = dataset[["TurbID", "Tmstamp", "Wspd", "Wdir", "Etmp", "Wspd_w", "Patv"]]
        dataset = dataset.loc[dataset.Tmstamp > '2021']
        
        # Get unique turbine IDs in dataset
        # dataset_turbines = sorted(dataset['TurbID'].unique())

        
        # Filter location to only include turbines in dataset
        # location = location[location.index.isin(dataset_turbines)]
        points = location[["x", "y"]].values
        
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # Create adjacency matrix from triangulation
        n_points = len(points)
        adjacency_matrix = np.zeros((n_points, n_points))
        
        # Mark connected points (edges) as 1
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = simplex[i], simplex[j]
                    adjacency_matrix[p1, p2] = 1
                    adjacency_matrix[p2, p1] = 1
        
        # Calculate Euclidean distance matrix
        euclidean_dist = squareform(pdist(points))
        
        # Create triangle-based distance matrix
        distance_matrix = np.full((n_points, n_points), np.inf)
        distance_matrix[adjacency_matrix == 1] = euclidean_dist[adjacency_matrix == 1]
        np.fill_diagonal(distance_matrix, 0)
        
        # Direct pivot_table with multi-level columns
        pivot_df = dataset.pivot_table(
            index='Tmstamp',
            columns='TurbID', 
            values=['Patv']
            # 'Wspd', 'Wdir', 'Etmp', 'Wspd_w', 
        )
        
        # Swap column levels to have TurbID as first level
        pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1)        
        
        return pivot_df, distance_matrix, tri.simplices
        
    def load(self, impute_zeros=True):
        df, dist, tri = self.load_raw()
        # fill NaN for 0
        df = df.fillna(0.)
        
        mask = (df.values != 0.).astype('uint8')

        # Get number of unique values at each level
        n_turbines = df.columns.get_level_values(0).nunique()
        n_channels = df.columns.get_level_values(1).nunique()

        mask = mask.reshape(mask.shape[0], n_turbines, n_channels)

        if impute_zeros:
            # replace 0 by forward fill
            df = df.ffill()

        return df, dist, mask, tri
        
    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
        elif method == "precomputed":
            return self.dist