from torch import nn
from torch_geometric.nn import fps, knn, max_pool_x

from ..types import PointSetBatchInfo


class TransitionDown(nn.Module):
    def __init__(self,
                 *,
                 in_features: int,
                 out_features: int,
                 num_neighbors: int,
                 fps_sample_ratio: float):
        super().__init__()
        self.fps_sample_ratio = fps_sample_ratio
        self.num_neighbors = num_neighbors
        self.mlp = nn.Sequential(nn.Linear(in_features, out_features),
                                 nn.BatchNorm1d(num_features=out_features),
                                 nn.ReLU()
                                 )

    def forward(self, input: PointSetBatchInfo) -> PointSetBatchInfo:
        """input contains:
            features [N x in_features] - node features
            positions [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch - batch indices
        """
        features, positions, batch = input
        fps_indices = fps(positions, batch=batch, ratio=self.fps_sample_ratio)
        out_features = self.mlp(features)

        # [2 x num_neighbors * fps_indices]
        nearthest_indices = knn(
            features, features[fps_indices], k=self.num_neighbors, batch_x=batch, batch_y=batch[fps_indices])

        new_features = max_pool_x(
            cluster=nearthest_indices[0, :], x=out_features[nearthest_indices[1, :]], batch=batch[nearthest_indices[1, :]])[0]

        return new_features, positions[fps_indices], batch[fps_indices]
